from .dbcore import session, taskModel, weightModel, hyperParam, \
    hyperParamLog, dataFormat, metricModel, metricLog, trainModel, logModel
from pathlib import Path
from datetime import datetime
import json, os
from .utils import create_dir
from .config import DATADIR

tsDict = {"created_at": datetime.now(), "updated_at": datetime.now()}
tsDict_ = {"updated_at": datetime.now()}


class forgedb(object):
    def __init__(self, task, remark="created_in_code", framewk="pytorch", verbose=True, log_hparam_read=False):
        """
        connect to a task, will create a new task if not already established
        :param task: task name string
        :param remark: Introduction about this task
        """
        super().__init__()
        self.s = session
        self.task = self.s.query(taskModel).filter(taskModel.taskname == task).first()
        self.verbose = verbose
        self.log_hparam_read = log_hparam_read
        if self.task == None:
            if self.verbose: print("[creating task:%s]" % (task))
            taskitem = taskModel(taskname=task, remark=remark,
                                 created_at=datetime.now(), updated_at=datetime.now())
            self.s.add(taskitem)
            self.s.flush()
            self.s.commit()
            self.task = taskitem
        self.taskdir = Path(os.path.join(DATADIR, self.task.taskname))
        self.weightdir = self.taskdir / "weights"
        self.logsdir = self.taskdir / "logs"
        create_dir(self.taskdir)
        create_dir(self.weightdir)
        create_dir(self.logsdir)
        self.hp2dict()
        if self.verbose:
            print("=" * 10 + "hyper params" + "=" * 10)
            print(self.confdict)
        self.framewk = framewk

        # check/update necessary formats
        self.format("float", "Float Format")
        self.format("int", "Integer Format")
        self.format("str", "String Format")

        self.set_hp_attributes()
        self.nb_trains = 0
        self.new_train()
        self.modelnow = self.new_model_name()

    def __repr__(self):
        return "[forge:%s]" % (self.task.taskname)

    def get_hyperparams(self):
        """
        from task to hyper parameters
        :return: a list of hyper params
        """
        return self.s.query(hyperParam).filter(hyperParam.task_id == self.task.id).all()

    def hp2dict(self, ):
        """
        :return: hplist, hpdict
        """
        hplist = self.get_hyperparams()
        self.confdict = dict((hp.slug, eval(hp.format.name)(hp.val)) for hp in hplist)
        return hplist, self.confdict

    def set_hp_attributes(self):
        list(setattr(self, hpslug, hpval) for hpslug, hpval in self.confdict.items())

    def p(self, key, val=None):
        """
        Access to hyper parameter
        :param key: parameter name/slug, avoid space or strange characters, letters and digits only
        :param val:
        * read value, default none to read value
        * if pass a kwarg here, will set the value to sql db
        :return: the parameter value
        """
        if val:
            hp = self.s.query(hyperParam).filter(hyperParam.slug == key, hyperParam.task_id == self.task.id).first()
            if hp:
                hp.val = str(val)
                hp.updated_at = datetime.now()

            else:
                fmt = self.get_format(val)
                hp = hyperParam(task_id=self.task.id,
                                slug=key,
                                remark="Created in task %s" % (self.task.taskname),
                                format_id=fmt.id, **tsDict, val=str(val))
            self.s.add(hp)
            self.s.commit()
            # write a log of new entry
            hplog = hyperParamLog(hp_id=hp.id, train_id=self.train.id, valsnap=hp.val, **tsDict)
            self.s.add(hplog)
            self.s.commit()
            return eval(hp.format.name)(hp.val)
        else:
            hp = self.s.query(hyperParam).filter(hyperParam.slug == key, hyperParam.task_id == self.task.id).first()
            if hp:
                if self.log_hparam_read:
                    hplog = hyperParamLog(hp_id=hp.id, train_id=self.train.id, valsnap=hp.val, **tsDict)
                    self.s.add(hplog)
                    self.s.commit()
                return eval(hp.format.name)(hp.val)

    def new_train(self, trainname=None, remark=None):
        """
        Add a training
        :param trainname:
        :param remark:
        :return:
        """
        if trainname == None: trainname = "%s-%s" % (self.task.taskname, datetime.now().strftime("%H:%M:%S %Y-%m-%d"))
        if remark == None: remark = "train was generated in code"
        train = trainModel(name=trainname, task_id=self.task.id, remark=remark, **tsDict)
        self.s.add(train)
        self.s.commit()
        self.train = train
        self.nb_trains += 1
        return train

    def format(self, name, remark=None):
        """
        get the format by name, like str
        , if not found,
        registering format
        :param obj: format class
        """
        fmt = self.s.query(dataFormat).filter(dataFormat.name == name).first()
        if fmt == None:
            fmt = dataFormat(name=name, remark=remark, )
            self.s.add(fmt)
            self.s.commit()
            print("[creating format :%s] for 1st and last time" % (name), flush=True)
        return fmt

    def get_format(self, val):
        """
        A sample value to return format object
        :param val: sample value
        :return: format object
        """
        typename = type(val).__name__
        return self.format(name=typename)

    def new_model_name(self, extra_name="model"):
        """
        :param extra_name: optional, default model, describe this in 1 consequtive string, something like model structure
        :return: a model name
        """
        self.modelnow = "%s_%s_%s" % (self.task.taskname, extra_name, self.train.id)
        return self.modelnow

    def save_weights(self, path, modelname=None, framewk=None):
        hplist, hpdict = self.hp2dict()
        if framewk:
            self.framewk = framewk
        mn = modelname if modelname else self.new_model_name()
        w = weightModel(task_id=self.task.id, name=mn,
                        path=str(path), framewk=self.framewk, train_id=self.train.id,
                        params_json=json.dumps(hpdict),
                        **tsDict,
                        )
        self.s.add(w)
        self.s.commit()
        return w

    def m_(self, key, val, big_better=True,
           remark=None, ):
        """
        Saving the metrics, create/ update metric value
        key: metric name
        val: metric value
        """
        val = str(val)
        mt = self.s.query(metricModel).filter(metricModel.slug == key, metricModel.task_id == self.task.id).first()
        if remark == None:
            remark = "creating from task:%s" % (self.task.taskname)
        if mt:
            mt.val = val
            mt.updated_at = datetime.now()
        else:
            fmt = self.get_format(val)
            mt = metricModel(slug=key,
                             task_id=self.task.id,
                             format_id=fmt.id,
                             val=str(val),
                             big_better=big_better,
                             remark=remark,
                             **tsDict)

        return mt

    def m(self, key, val, epoch=0, big_better=True,
          remark=None, ):
        """
        Saving the metrics, with the logs
        key: metric name
        val: metric value
        big_better, is this metric the bigger the better (for the model)
        """
        # Saving the current metric
        mt = self.m_(key, val, big_better=big_better, remark=remark)  # update or new
        self.s.add(mt)
        self.s.commit()
        # Saving the metric log
        mlog = metricLog(metric_id=mt.id, train_id=self.train.id, epoch=epoch, valsnap=str(val), **tsDict)
        self.s.add(mlog)
        self.s.commit()
        return mt

    def save_metrics(self, metrics, epoch=0, small_list=None):  # todo improve small list
        """
        saving a dictionary of metrics
        :param metrics: dictionary
        :param small_list: a list of metric names, the smaller value represent better model
        :return:
        """
        if small_list == None: small_list = []
        mtlist = []
        for k, v in metrics.items():
            kwa = dict({"key": k, "val": v, "epoch": epoch, })
            if k in small_list:
                kwa.update({"big_better": False})
            self.m(**kwa, )

    def log_record(self, path):
        l = logModel(path=path, train_id=self.train.id, task_id=self.task.id, **tsDict)
        self.s.add(l)
        self.s.commit()
        if self.verbose: print("[log saved to]:%s" % (path), flush=True)
        return l

    def savejson(self, path=None):
        hplist, hpdict = self.hp2dict()
        conf_dict = dict({"hp": hpdict, "taskname": self.task.taskname})
        if path == None:
            path = str(self.taskdir / str("conf_%s_%s.json" % (self.task.taskname, datetime.now().timestamp())))
        with open(path, 'w') as outfile:
            json.dump(conf_dict, outfile)
        print("configuration saved to %s" % (path))

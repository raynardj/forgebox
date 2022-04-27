__all__ = ['Opts']


class Opts(object):
    def __init__(self, *args, **kwargs):
        """
        opts = Opts(opt1 = opt1, opt2 = opt2)
        opts.opt2.zero_grad()
        opts["opt3"] = opt3
        print(len(opts))
        """
        self.optlist = []
        self.optnames = []
        for i in range(len(args)):
            oname = f"optimizer_no{i + 1}"
            setattr(self, oname, args[i])
            self.optlist.append(args[i])
            self.optnames.append(oname)
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.optlist.append(v)
            self.optnames.append(k)

    def __repr__(self):
        return "\n".join(list(
            f"{self.optnames[i]}\n\t{self.optlist[i].__class__}\n\t{self.read_opt(self.optlist[i])}" for i in
            range(len(self.optnames))))

    def get_pg(self, opt):
        """
        Get paramgroups dictionary, informations about an optimizer
        opt:torch.optim.optimizer
        """
        return dict.copy(opt.param_groups[0])

    def read_opt(self, opt):
        rt = self.get_pg(opt)
        if "params" in rt:
            del rt["params"]
        return rt

    def __len__(self):
        """
        Total number of optimizers
        """
        return len(self.optlist)

    def __contains__(self, item):
        return item in self.optlist

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, optimizer):
        self.optlist.append(optimizer)
        self.optnames.append(key)
        setattr(self, key, optimizer)

    def zero_all(self):
        """
        Zero gradient on all the optimizers
        """
        for opt in self.optlist:
            opt.zero_grad()

    def step_all(self):
        """
        All the optimizers match a step
        """
        for opt in self.optlist:
            opt.step()
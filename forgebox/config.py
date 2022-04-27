import json


class Config(dict):
    def __setattr__(self,k,v):
        self[k]=v

    def __getattr__(self,k,):
        return self[k]

    def __delattr__(self,k):
        del self[k]

    def pretty_print(self):
        print(json.dumps(self,indent = 4))

    def __call__(self,**kwargs):
        """
        assign more keyword value
        """
        self.update(kwargs)
        return self

    def save(self,json_path,indent=None):
        """
        save to json file
        """
        with open(json_path,"w") as f:
            json.dump(self,f,indent = indent)

    @classmethod
    def load(cls,path):
        """
        load from json file
        """
        with open(path,"r") as f:
            obj = cls()(**json.loads(f.read()))
        return obj

    def first(self,key):
        return first(self,key)

    def getall(self,key):
        return getall(self,key)

def first(d,key):
    if hasattr(d,"items"):
        for k,v in d.items():
            if k==key: return v
            else:
                ans = first(v,key)
                if ans!=None: return ans

    if type(d) in [tuple,list,set]:
        for i in d:
            ans = first(i,key)
            if ans!=None: return ans
    return None

def getall(d,key):
    results = []
    if hasattr(d,"items"):
        for k,v in d.items():
            if k==key: results+=[v,]
            else: results += getall(v,key)

    if type(d) in [tuple,list,set]:
        for i in d:
            results += getall(i,key)
    return results

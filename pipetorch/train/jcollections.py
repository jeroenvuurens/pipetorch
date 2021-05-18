from collections import defaultdict

class attributedict(dict):
    __getattr__ = dict.__getitem__

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        __setattr__ = attributedict.__setitem__

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        
    def clone(self):
        return attributedict(**self)
     
class config(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self._hooks = defaultdict(set)
        #__setattr__ = config.__setitem__

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __setitem__(self, key, value):
        oldvalue = self.get(key, None)
        if value != oldvalue:
            dict.__setitem__(self, key, value)
            for f in self._hooks[key]:
                f(key, oldvalue, value)

    def update(self, kwargs):
        for k,v in kwargs.items():
            self[k] = v

    def remove(self, key):
        try:
            del self[key]
        except: pass

    def add_hook(self, f, *keys):
        for key in keys:
            self._hooks[key].add(f)
       
    def del_hook(self, f, *keys):
        for key in keys:
            self._hooks[key].remove(f)

    def clone(self):
        return config(**self)




import pandas as pd
import matplotlib.pyplot as plt

class Hashabledict(dict):
    def __hash__(self):
        return hash((frozenset(self), frozenset(self.values())))
    
class Hyperparameters:
    def __init__(self, **parameters):
        self.orig_params = parameters
        self.params = self._parameters()
        self.results = {}
    
    def _run(self, p):
        if p not in self.results:
            r = self.run(**p)
            r.update(p)
            self.results[p] = r
    
    def grid(self):
        for p in self.params:
            self._run(p)
   
    def _parameters(self):
        param = [(p, r, iter(r)) for p, r in self.orig_params.items() ]
        param = [(p, r, i, next(i)) for p, r, i in param ]
        tt = []
        while len(param) > 0:
            tt.append( Hashabledict({ p:v for p, _, _, v in param }) )
            for j in range(len(param)):
                p, r, i, v = param[j]
                v = next(i, None)
                if v != None:
                    param[j] = (p, r, i, v)
                    break
                if j == len(param)-1:
                    return tt
                i = iter(r)
                v = next(i)
                param[j] = (p, r, i, v)

    def plot(self, param, metric):
        coords = []
        for paramvalue in { p[param] for p in self.results }:
            results = [ r[metric] for p, r in self.results.items() if p[param] == paramvalue ]
            coords.append((paramvalue, sum(results)/len(results)))
        x, y = zip(*sorted(coords))
        plt.plot(x, y)
        
    def plot2(self, param1, param2, metric):
        coords = []
        for paramvalue1 in { p[param1] for p in self.results }:
            for paramvalue2 in { p[param2] for p in self.results }:
                results = [ r[metric] for p, r in self.results.items() if p[param1] == paramvalue1 and p[param2] == paramvalue2 ]
                coords.append((paramvalue1, paramvalue2, sum(results)/len(results)))
        df = pd.DataFrame(coords, columns=(param1, param2, metric))
        df.plot(kind='scatter', x=param1, y=param2, c=metric, sharex=False, cmap=plt.get_cmap("jet"))

    def db(self, **params):
        try:
            if self._params == params:
                return self._db
            print(f'new db {params}')
        except: pass
        self._params = params
        self._db = self.create_db(**params)
        return self._db

    @staticmethod
    def create_db(self, **params):
        raise NotImplementedError('You have to define create_db(**params)')

    def run(self, **params):
        raise NotImplementedError('You have to define run(**params)')



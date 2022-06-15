from functools import wraps

def is_documented_by(original, replace={}):
    @wraps(original)
    def wrapper(target):
        docstring = original.__doc__
        for key, value in replace.items():
            docstring = docstring.replace(key, value)
        target.__doc__ = docstring
        return target
    return wrapper

def run_magic(magic, line, cell=None):
    from IPython import get_ipython
    ipython = get_ipython()
    if cell is None:
        ipython.run_line_magic(magic, line)
    else:
        ipython.run_cell_magic(magic, line, cell)

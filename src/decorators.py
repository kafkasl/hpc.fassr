import numpy as np
from settings.basic import logging

def nan_on_exception(func):
    def wrapper(*args, **kwargs):
        res = np.nan
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            print("NaN on Exception [%s] computing: %s(%s, %s)" %
                          (e, func.__name__, args, kwargs))
        return res
    return wrapper
import time
from functools import wraps

def timeit(name=None):
    def decorator(fn):
        label = name or fn.__name__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[TIME] {label}: {dt:.2f} ms")
            return out
        return wrapper
    return decorator


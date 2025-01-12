from functools import wraps
import logging
import time

def my_logger(orig_func):
    # Create a logger with the function name
    logging.basicConfig(
        filename=f'{orig_func.__name__}.log',
        level=logging.INFO
    )
    
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            f'Running {orig_func.__name__} with args: {args}, kwargs: {kwargs}'
        )
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print(f'{orig_func.__name__} ran in: {t2:.2f} sec')
        return result
    return wrapper
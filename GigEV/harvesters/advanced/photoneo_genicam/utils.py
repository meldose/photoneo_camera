import time # imported time module
from functools import wraps # imported functools module


def measure_time(func): # defined the fucnction measure_time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time() # assigning the value of the time.time() to start_time
        result = func(*args, **kwargs)
        end_time = time.time() # assigning the value of the time.time() to end_time
        execution_time_ms = (end_time - start_time) * 1000 # execution_time_ms = end_time - start_time * 1000
        print(f"Execution time of {func.__name__}: {execution_time_ms:.2f} milliseconds")
        return result

    return wrapper # returning the wrapper 


def write_raw_array(filename: str, array): # defined the function write_raw_array
    with open(filename, "wb") as file:
        for element in array: # checking the element in the array
            file.write(element.tobytes()) # write the element in the array

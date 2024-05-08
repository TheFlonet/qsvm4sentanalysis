import time
from typing import Callable, List, Dict


def eval_time(func: Callable) -> Callable:
    def inner(*args: List, **kwargs: Dict) -> None:
        st, stp = time.time(), time.process_time()
        func(*args, **kwargs)
        et, etp = time.time(), time.process_time()
        print('Execution time:', et - st)
        print('Process time:', etp - stp)
        print('-' * 100)

    return inner

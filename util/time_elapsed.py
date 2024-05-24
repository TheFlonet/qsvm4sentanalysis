import logging
import time
from typing import Callable, List, Dict

log = logging.getLogger('qsvm')


def eval_time(func: Callable) -> Callable:
    def inner(*args: List, **kwargs: Dict) -> None:
        st, stp = time.time(), time.process_time()
        func(*args, **kwargs)
        et, etp = time.time(), time.process_time()
        log.info(f'Execution time: {et- st}')
        log.info(f'Process time: {etp - stp}')
    return inner

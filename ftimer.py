import time

class Ftimer(object):
    def __init__(self, name, logging=False):
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self.logging = logging

    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        elapsed_time = (self.end_time - self.start_time) * 1000
        if self.logging:
            from loguru import logger
            logger.info(f"{self.name} time: {elapsed_time:.2f} ms.")
        else:
            print(f"{self.name} time: {elapsed_time:.2f} ms.")
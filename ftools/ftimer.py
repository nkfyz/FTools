import time

class Ftimer(object):
    """Timer class for measuring the time of a code segment as a context manager.

    Keyword Arguments:
        - name: str: Name of the timer.
        - logging: bool: Whether using the logging format to print elapsed time.
    """
    def __init__(self,
                 name="Code",
                 type="INFO",
                 logging=False,
    ):
        self.name = name
        self.type = type
        self.start_time = 0
        self.end_time = 0
        
        self.logger = None
        self.logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>"+ self.type + "</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        self.logging = logging
        
        if self.logging:
            from loguru import logger
            import sys
            self.logger = logger
            self.logger.remove()
            self.logger.add(sys.stderr, format=self.logger_format)
            
    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        elapsed_time = (self.end_time - self.start_time) * 1000
        if self.logging:
            self.logger.opt(depth=1).info(f"{self.name} time: {elapsed_time:.2f} ms.")
        else:
            print(f"{self.name} time: {elapsed_time:.2f} ms.")
    
    """ return ms """
    def get_elapsed_time(self):
        return (self.end_time - self.start_time) * 1000
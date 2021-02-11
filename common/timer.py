import datetime


class Timer:
    def __init__(self):
        self.start = None
        self.final = None

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.final = datetime.datetime.now() - self.start

    def total(self):
        if self.final is None:
            raise RuntimeError('Timer total called before exit start={}'.format(self.start))
        return self.final

    def elapsed(self):
        if self.start is None:
            raise RuntimeError('Timer elapsed called before start')
        return datetime.datetime.now() - self.start

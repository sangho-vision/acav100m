import random


'''
code from https://github.com/tmbdev/webdataset/blob/master/webdataset/filters.py
'''


def shuffle_(data, bufsize=1000, initial=100, rng=random):
    """Shuffle the data in the stream.
    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.
    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance
    """
    initial = min(initial, bufsize)
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
            except StopIteration:
                pass
        k = rng.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


class Curried2(object):
    """Helper class for currying pipeline stages.
    We use this roundabout construct becauce it can be pickled."""

    def __init__(self, f, *args, **kw):
        self.f = f
        self.args = args
        self.kw = kw

    def __call__(self, data):
        return self.f(data, *self.args, **self.kw)

    def __str__(self):
        return f"<{self.f.__name__} {self.args} {self.kw}>"

    def __repr__(self):
        return f"<{self.f.__name__} {self.args} {self.kw}>"


class Curried(object):
    """Helper class for currying pipeline stages.
    We use this roundabout construct becauce it can be pickled."""

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kw):
        return Curried2(self.f, *args, **kw)


shuffle = Curried(shuffle_)

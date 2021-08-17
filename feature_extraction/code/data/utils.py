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

    We use this roundabout construct because it can be pickled."""

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kw):
        return Curried2(self.f, *args, **kw)

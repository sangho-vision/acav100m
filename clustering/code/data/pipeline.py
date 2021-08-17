from functools import reduce


def compose2(f, g):
    """Compose two functions, g(f(x))"""
    return lambda x: g(f(x))


def compose(*args):
    """Compose a sequence of functions (left-to-right)"""
    return reduce(compose2, args)


def pipeline(source, *args):
    """Write an input pipeline; first argument is source, rest are filters."""
    if len(args) == 0:
        return source
    return compose(*args)(source)

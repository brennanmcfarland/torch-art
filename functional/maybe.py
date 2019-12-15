from functional.core import pipe as core_pipe


# maybe monad bind operation
def bind(f, x):
    return None if x is None else f(x)


# maybe monad pipe operation
def pipe(*fs):
    return core_pipe([lambda x: bind(f, x) for f in fs])


# pipe until not none
def pipe_until(*fs):
    return core_pipe([lambda x: not bind(f, x) for f in fs])

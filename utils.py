

# reduces a list to it's item if there's only one, else returns the list unchanged
def try_reduce_list(l):
    l = list(l)
    if len(l) == 1:
        return l[0]
    else:
        return l


# hook is the specific point at which the callbacks are being called, eg on_step
def run_callbacks(hook, callbacks, *args):
    results = []
    for callback in callbacks:
        if hook in callback:
            results.append(callback[hook](*args))
    return results

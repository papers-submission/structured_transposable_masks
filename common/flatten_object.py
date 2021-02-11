def flatten_object(obj, delimiter='.', prefix=''):
    def flatten(x, name=prefix):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + delimiter)
        elif isinstance(x, list) or isinstance(x, tuple):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + delimiter)
        else:
            out[name[:-1]] = x

    out = {}
    flatten(obj)
    return out

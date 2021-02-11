import json


def json_is_serializable(obj):
    serializable = True
    try:
        json.dumps(obj)
    except TypeError:
        serializable = False
    return serializable


def json_force_serializable(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = json_force_serializable(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = json_force_serializable(v)
    elif isinstance(obj, tuple):
        obj = list(obj)
        for i, v in enumerate(obj):
            obj[i] = json_force_serializable(v)
        obj = tuple(obj)
    elif not json_is_serializable(obj):
        obj = 'filtered by json'
    return obj

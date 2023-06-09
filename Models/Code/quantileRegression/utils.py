import json
import inspect


def make_serializable(params: dict) -> dict:
    params = {**params}  # Make a copy
    for key in params:
        if type(params[key]) is dict:
            params[key] = make_serializable(params[key])
        else:
            try:
                json.dumps(params[key])
            except (TypeError, OverflowError):
                if key == "f":
                    funcString = str(inspect.getsourcelines(params[key])[0])
                    params[key] = funcString.strip("['\\n'],").split('"f": ')[1]
                else:
                    params[key] = params[key].__class__.__name__
    return params

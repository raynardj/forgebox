from typing import Tuple, Any


class Unpack:
    """
    Simmulat js unpacking variables
    Assume you have very convoluted data structure
    data = {
        "info":{
            "name": "Lisa", "age":12,
            },
        "grad":{
            "math":{
                "mid":98, "final":99
                },
            "history":{"mid":90, "final":95},
            },
        "date":"2022-01-01",
        }
    student = Unpack(data)
    name, final_math_score, date = student(["info","name"],['grad','math','final'],'date')
    """

    def __init__(self, obj, raise_error: bool = False):
        self.obj = obj
        self.raise_error = raise_error

    def indexing(self, key: str) -> Any:
        try:
            return self.obj[key]
        except (KeyError, IndexError) as e:
            if self.raise_error:
                raise e
            else:
                return None

    def __call__(self, *args) -> Tuple[Any]:
        rt = []
        for arg in args:
            if type(arg) in [list, tuple]:  # arg is iterable
                if len(arg) > 1:
                    answer = self.indexing(arg[0])
                    if type(answer) in [dict, list, tuple]:
                        # answer has further indexing possibilities
                        rt.append(Unpack(answer)(arg[1:]))
                    else:
                        # still remaining keychains, but no tree branch to go on
                        rt.append(answer)
                elif len(arg) == 1:
                    rt.append(self.indexing(arg[0]))
                else:
                    if self.raise_error:
                        raise ValueError(f"list length can not equal to zero")
                    else:
                        rt.append(None)
            else:  # arg is just one key
                rt.append(self.indexing(arg))

        if len(rt) == 1:
            return rt[0]
        else:
            return tuple(rt)

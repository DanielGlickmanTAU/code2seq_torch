from typing import Dict, List
from itertools import product


def gridsearch(default_params: Dict, params_to_grid_search: Dict[object, List]) -> List[Dict]:
    def prod(params_to_grid_search):
        values = []
        for v in params_to_grid_search.values():
            if isinstance(v, list):
                values.append(v)
            elif isinstance(v, str):
                values.append([f'"{v}"'])
            else:
                values.append([v])
        return product(*values)

    def flatten_tuples(d):
        ret = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                k_1, k_2 = k
                v_1, v_2 = v
                ret[k_1] = v_1
                ret[k_2] = v_2
            else:
                ret[k] = v
        return ret

    params_as_dicts = [dict(zip(params_to_grid_search.keys(), v)) for v in prod(params_to_grid_search)]
    params_as_dicts = [flatten_tuples(d) for d in params_as_dicts]

    rets = []
    for d in params_as_dicts:
        ret = default_params.copy()
        ret.update(d)
        rets.append(ret)
    return rets

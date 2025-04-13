import gc


def cache_clear(current_vars, _keep_vars=None):
    for var in list(current_vars):
        if (_keep_vars is None or var not in _keep_vars) and not var.startswith("_"):
            del current_vars[var]

    gc.collect()

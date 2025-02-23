import lithops

import virtualizarr

fexec = lithops.FunctionExecutor(config_file="lithops.yaml")


def test_function(args: dict):
    return virtualizarr.__version__


result = fexec.call_async(func=test_function, data={"args": {}}).result()
print(result)

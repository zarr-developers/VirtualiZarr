import icechunk
import lithops

fexec = lithops.FunctionExecutor(config_file="lithops.yaml")


def test_function(args: dict):
    print(icechunk.__version__)


result = fexec.call_async(func=test_function, data={"args": {}})  # .get_result()
print(result)

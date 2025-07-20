We define a Parser using the following [Protocol][typing.Protocol] so that external implementations can
independently develop implementations that work with [virtualizarr.open_virtual_dataset][] and [virtualizarr.open_virtual_mfdataset][].

You can add configuration options to your Parser implementation's `__init__` method.

::: virtualizarr.parsers.Parser

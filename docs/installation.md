# Installation

Currently you need to clone VirtualiZarr and install it locally:

```shell
git clone https://github.com/TomNicholas/VirtualiZarr
cd VirtualiZarr
pip install -e .
```

You will also need a specific branch of xarray in order for concatenation without indexes to work. (See [this comment](https://github.com/TomNicholas/VirtualiZarr/issues/14#issuecomment-2018369470).)


## Install Test Dependencies

```shell
pip install '-e .[test]'
```


## Install Docs Dependencies

```shell
pip install '-e .[docs]'
```

## Docker

Build the runtime with 

```bash
lithops runtime build -b aws_lambda -f ./docker/Dockerfile virtualizarr-runtime
```

this might require you to edit the lithops source to add the flags `--provenance=false --no-cache` to the `docker build` command. (TODO: test removing these)

Then deploy it with

```bash
lithops runtime deploy -b aws_lambda virtualizarr-runtime
```

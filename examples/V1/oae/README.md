## Docker
Build the runtime with

```bash
lithops runtime build -b aws_lambda -f ./docker/Dockerfile virtualizarr-runtime
```

Then deploy it with

```bash
lithops runtime deploy -b aws_lambda virtualizarr-runtime
```

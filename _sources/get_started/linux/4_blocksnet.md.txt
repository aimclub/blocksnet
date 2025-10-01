# Install BlocksNet

The installation differs depending on the purpose.

## For using in own project

If you cloned your own repository.

1. Install the required BlocksNet version and extras:
   ```bash
   pip install blocksnet[full]
   ```
   And get the following result:
   ```bash
   Successfully installed blocksnet-1.0.0
   ```
2. Add BlocksNet to your `requirements.txt`:
   ```
   # requirements.txt
   ...
   blocksnet==1.0.0
   ...
   ```
   Or `pyproject.toml`:
   ```toml
   dependencies=[
    ...
    "blocksnet[full]==1.0.0",
   ]
   ```

## For contributing

If you cloned BlocksNet repository for participating in development just install the BlocksNet with required `dev` packages:

```bash
pip install -e ".[dev]"
```

And get the following result:

```bash
Successfully installed blocksnet-1.0.0.post4
```

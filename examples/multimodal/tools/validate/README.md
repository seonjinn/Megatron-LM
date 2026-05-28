# Megatron Dataloader Validator

This tool can validate that an energon dataset can be accessed properly. I.e. the first sample of each dataset can be read.

* It uses the cookers, so those are verified as well.
* It does **not** verify that the task encoder is working. For that, use the `iter_data.py` script (requires a gpu).

Run:
```sh
uv sync
uv run validate-dataset <path to dataset.yaml>
```

If it prints
```txt
[...]
Verified all datasets
Total samples of datasets: 1000000
Total samples loaded: 10
```

you're likely good.
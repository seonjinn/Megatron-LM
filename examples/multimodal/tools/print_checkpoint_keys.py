"""Print all keys from a checkpoint directory in dot notation.

Supports three checkpoint formats:
  1. Original Megatron (multi-TP/PP): mp_rank_XX_YYY/model_optim_rng.pt
  2. TP1 Megatron: mp_rank_00/model_optim_rng.pt
  3. HF safetensors: model-XXXXX-of-YYYYY.safetensors (or via model.safetensors.index.json)

Usage:
  python print_checkpoint_keys.py <checkpoint_dir> [--output <file>] [--rank <rank_dir>] [--section model]
"""

import argparse
import builtins
import glob
import json
import os
import sys
import types


class _StubModule(types.ModuleType):
    """Module that returns stub objects for any attribute access."""

    def __getattr__(self, name):
        return _StubClass

    def __call__(self, *args, **kwargs):
        return _StubClass()


class _StubClass:
    """Catchall class that can be instantiated/called/subscripted freely."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _StubClass()

    def __getattr__(self, name):
        return _StubClass

    def __class_getitem__(cls, item):
        return cls

    def __reduce__(self):
        return (_StubClass, ())


class stub_missing_modules:
    """Context manager: patches builtins.__import__ to stub missing modules.

    Pickle's C-level Unpickler.find_class calls __import__ directly, bypassing
    sys.meta_path finders. Patching builtins.__import__ is the only reliable way
    to intercept those lookups.
    """

    COMMON_PREFIXES = ["megatron", "apex", "transformer_engine", "nemo"]

    def __init__(self, extra_prefixes=None):
        self.prefixes = self.COMMON_PREFIXES + (extra_prefixes or [])
        self._original_import = None

    def _matches(self, name):
        for prefix in self.prefixes:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False

    def __enter__(self):
        self._original_import = builtins.__import__
        ctx = self

        def _patched_import(name, *args, **kwargs):
            if ctx._matches(name):
                if name not in sys.modules:
                    mod = _StubModule(name)
                    mod.__path__ = []
                    sys.modules[name] = mod
                return sys.modules[name]
            return ctx._original_import(name, *args, **kwargs)

        builtins.__import__ = _patched_import
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._original_import
        to_remove = [k for k in sys.modules
                     if isinstance(sys.modules[k], _StubModule)]
        for m in to_remove:
            del sys.modules[m]
        return False


def collect_keys_recursive(obj, prefix=""):
    """Recursively collect keys from nested dicts/lists using dot notation."""
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            child_keys = collect_keys_recursive(v, full_key)
            if child_keys:
                keys.extend(child_keys)
            else:
                keys.append(full_key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj, ):
            full_key = f"{prefix}[{i}]"
            child_keys = collect_keys_recursive(v, full_key)
            if child_keys:
                keys.extend(child_keys)
            else:
                keys.append(full_key)
    else:
        return []
    return keys


def get_tensor_shape_str(val):
    """Return shape string for tensor-like objects, empty string otherwise."""
    if hasattr(val, 'shape'):
        return f"  shape={list(val.shape)}"
    return ""


def load_torch_checkpoint_keys(pt_path, section=None, show_shapes=False):
    """Load a .pt checkpoint and return sorted keys in dot notation."""
    import torch
    with stub_missing_modules():
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    if section:
        if section not in ckpt:
            print(f"Warning: section '{section}' not found. Available: {list(ckpt.keys())}")
            return []
        obj = ckpt[section]
        prefix = section
    else:
        obj = ckpt
        prefix = ""

    if isinstance(obj, dict) and not show_shapes:
        keys = collect_keys_recursive(obj, prefix)
    elif isinstance(obj, dict) and show_shapes:
        keys = []
        flat = flatten_dict(obj, prefix)
        for k, v in sorted(flat.items()):
            keys.append(k + get_tensor_shape_str(v))
    else:
        keys = [prefix] if prefix else [str(type(obj))]

    return sorted(keys)


def flatten_dict(d, prefix=""):
    """Flatten a nested dict into {dotted_key: value} pairs."""
    items = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, full_key))
        else:
            items[full_key] = v
    return items


def load_safetensors_keys(directory, show_shapes=False):
    """Load keys from safetensors files in a directory."""
    index_path = os.path.join(directory, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        keys = sorted(index.get("weight_map", {}).keys())
        if show_shapes and keys:
            try:
                from safetensors import safe_open
                shape_map = {}
                shard_files = set(index["weight_map"].values())
                for shard in sorted(shard_files):
                    shard_path = os.path.join(directory, shard)
                    with safe_open(shard_path, framework="pt") as f:
                        for k in f.keys():
                            shape_map[k] = list(f.get_tensor(k).shape)
                keys = [f"{k}  shape={shape_map[k]}" if k in shape_map else k for k in keys]
            except ImportError:
                pass
        return keys

    st_files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not st_files:
        return []

    try:
        from safetensors import safe_open
    except ImportError:
        print("Error: safetensors package not installed. Install with: pip install safetensors")
        sys.exit(1)

    keys = set()
    for st_file in st_files:
        with safe_open(st_file, framework="pt") as f:
            if show_shapes:
                for k in f.keys():
                    keys.add(f"{k}  shape={list(f.get_tensor(k).shape)}")
            else:
                keys.update(f.keys())
    return sorted(keys)


def detect_format(directory):
    """Detect checkpoint format and return (format_name, relevant_paths)."""
    # Check for HF safetensors
    st_files = glob.glob(os.path.join(directory, "*.safetensors"))
    index_file = os.path.join(directory, "model.safetensors.index.json")
    if st_files or os.path.exists(index_file):
        return "safetensors", directory

    # Check for nested HF dir (e.g. mcore_to_hf/)
    mcore_hf = os.path.join(directory, "mcore_to_hf")
    if os.path.isdir(mcore_hf):
        st_check = glob.glob(os.path.join(mcore_hf, "*.safetensors"))
        if st_check:
            return "safetensors", mcore_hf

    # Check for single mp_rank_00/model_optim_rng.pt (TP1)
    single_rank = os.path.join(directory, "mp_rank_00", "model_optim_rng.pt")
    if os.path.exists(single_rank):
        return "torch", single_rank

    # Check for multi-rank mp_rank_XX_YYY dirs
    rank_dirs = sorted(glob.glob(os.path.join(directory, "mp_rank_*")))
    if rank_dirs:
        for rd in rank_dirs:
            pt_file = os.path.join(rd, "model_optim_rng.pt")
            if os.path.exists(pt_file):
                return "torch_multi", rank_dirs

    # Check if directory itself contains a .pt file
    pt_files = glob.glob(os.path.join(directory, "*.pt"))
    if pt_files:
        return "torch", pt_files[0]

    return "unknown", None


def main():
    parser = argparse.ArgumentParser(description="Print checkpoint keys in dot notation")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--section", "-s", default="model",
                        help="Section of torch checkpoint to print (default: 'model'). "
                             "Use 'all' to print everything.")
    parser.add_argument("--rank", "-r", default=None,
                        help="Specific rank directory for multi-rank checkpoints (default: first rank)")
    parser.add_argument("--shapes", action="store_true", help="Show tensor shapes")
    parser.add_argument("--all-ranks", action="store_true",
                        help="Print keys from all ranks (multi-rank only), prefixed with rank dir name")
    args = parser.parse_args()

    directory = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    section = None if args.section == "all" else args.section
    fmt, paths = detect_format(directory)

    print(f"Detected format: {fmt}", file=sys.stderr)
    print(f"Path: {directory}", file=sys.stderr)

    keys = []

    if fmt == "safetensors":
        keys = load_safetensors_keys(paths, show_shapes=args.shapes)
        print(f"Found {len(keys)} keys from safetensors", file=sys.stderr)

    elif fmt == "torch":
        print(f"Loading: {paths}", file=sys.stderr)
        keys = load_torch_checkpoint_keys(paths, section=section, show_shapes=args.shapes)
        print(f"Found {len(keys)} keys", file=sys.stderr)

    elif fmt == "torch_multi":
        rank_dirs = paths
        if args.all_ranks:
            for rd in rank_dirs:
                pt_file = os.path.join(rd, "model_optim_rng.pt")
                if os.path.exists(pt_file):
                    rank_name = os.path.basename(rd)
                    print(f"Loading: {pt_file}", file=sys.stderr)
                    rkeys = load_torch_checkpoint_keys(pt_file, section=section, show_shapes=args.shapes)
                    keys.extend(f"{rank_name}.{k}" for k in rkeys)
            print(f"Found {len(keys)} keys across {len(rank_dirs)} ranks", file=sys.stderr)
        else:
            if args.rank:
                target = os.path.join(directory, args.rank, "model_optim_rng.pt")
            else:
                target = os.path.join(rank_dirs[0], "model_optim_rng.pt")
            print(f"Loading: {target}", file=sys.stderr)
            keys = load_torch_checkpoint_keys(target, section=section, show_shapes=args.shapes)
            print(f"Found {len(keys)} keys from {os.path.basename(os.path.dirname(target))}", file=sys.stderr)
    else:
        print(f"Error: Could not detect checkpoint format in {directory}")
        sys.exit(1)

    output = "\n".join(keys)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Wrote {len(keys)} keys to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

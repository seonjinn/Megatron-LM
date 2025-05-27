import argparse
import json
import os
import re
from typing import List

from evaluate_mmmu import get_input_output_paths  # reuse helper

from sympy import sympify
# Import latex2sympy2; handle both old and new package layouts.
try:
    # Newer forks install as `latex2sympy2`; the original layout had a sub-package.
    from latex2sympy.latex2sympy2 import latex2sympy, Latex2SympyParsingError  # type: ignore
except ModuleNotFoundError:
    from latex2sympy2 import latex2sympy  # type: ignore
    # Some builds do not expose the specialised parsing error – fall back to a generic one.
    try:
        from latex2sympy2 import Latex2SympyParsingError  # type: ignore
    except ImportError:
        class Latex2SympyParsingError(Exception):
            """Fallback when upstream package does not export the parsing error."""
            pass

# -----------------------------------------------------------------------------
#  Helper functions that mimic the normalisation / grading used by the public
#  ValS-AI Math500 leaderboard (largely the same as PRM-800K grader).
# -----------------------------------------------------------------------------

_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_INLINE_MATH_RE = re.compile(r"\$([^$]+?)\$|\\\((.+?)\\\)")

# simple substitutions
_SUBS = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot|\\times", "*"),
    (r"\\dfrac|\\tfrac", "\\frac"),
    (r"\\pi|π", "pi"),
]


def _extract_final(text: str) -> str:
    """Get the segment that the leaderboard treats as the final answer."""
    # prefer last \boxed{...}
    boxed = _BOX_RE.findall(text)
    if boxed:
        return boxed[-1]

    # else last inline math span
    spans = _INLINE_MATH_RE.findall(text)
    if spans:
        # each tuple has two groups, only one non-empty
        last = spans[-1][0] or spans[-1][1]
        return last

    # fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _normalize(ans: str) -> str:
    ans = ans.strip().rstrip(".,;:!").lower()
    for pat, rep in _SUBS:
        ans = re.sub(pat, rep, ans)
    # drop all spaces
    ans = re.sub(r"\s+", "", ans)
    # outer parentheses or \( \)
    if ans.startswith("\\(") and ans.endswith("\\)"):
        ans = ans[2:-2]
    if ans.startswith("(") and ans.endswith(")"):
        ans = ans[1:-1]
    return ans


def _equiv(a: str, b: str) -> bool:
    """Sympy equivalence with graceful fallback to string match."""
    try:
        return latex2sympy(a) == latex2sympy(b)
    except (Latex2SympyParsingError, Exception):
        try:
            return sympify(a) == sympify(b)
        except Exception:
            return a == b


def compute_accuracy(result_file: str) -> float:
    """Compute exact-match accuracy over Math500 results (official style)."""
    data: List[dict] = json.load(open(result_file))
    correct = 0
    for idx, item in enumerate(data):
        # prediction key
        pred_raw = item.get("answer") or item.get("prediction") or item.get("predict") or ""
        pred_raw = _extract_final(pred_raw)
        pred = _normalize(pred_raw)

        gt_list = item.get("gt_answer", item.get("ground_truth", []))
        if isinstance(gt_list, str):
            gt_list = [gt_list]
        gt_list = [_normalize(_extract_final(g)) for g in gt_list]

        print(idx, 'PRED:', pred)
        print(idx, 'GT_LIST:', gt_list)
        if any(_equiv(pred, gt) for gt in gt_list):
            correct += 1
    return correct / len(data) * 100.0


def math500_eval(input_path) -> float:
    input_files, merged_out = get_input_output_paths(input_path, "Math500")

    # merge jsonl inputs into list
    all_results = {}
    for fp in input_files:
        with open(fp) as f:
            for line in f:
                obj = json.loads(line)
                sid = obj["sample_id"]
                if sid in all_results:
                    continue
                all_results[sid] = obj
    merged_list = list(all_results.values())

    with open(merged_out, "w") as f:
        json.dump(merged_list, f)

    acc = compute_accuracy(merged_out)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    args = parser.parse_args()

    acc = math500_eval(args.input_path)
    print(f"===== Math500 accuracy: {acc:.2f} =====")

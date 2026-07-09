import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGUMENTS_PATH = REPO_ROOT / "megatron" / "training" / "arguments.py"
TRAINING_PATH = REPO_ROOT / "megatron" / "training" / "training.py"


class SFTMetricAliasIntegrationTest(unittest.TestCase):
    def test_flag_is_disabled_by_default(self):
        tree = ast.parse(ARGUMENTS_PATH.read_text(encoding="utf-8"))
        matching_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first_arg = node.args[0]
            if (
                isinstance(first_arg, ast.Constant)
                and first_arg.value == "--log-comparison-metrics"
            ):
                matching_calls.append(node)

        self.assertEqual(len(matching_calls), 1)
        keywords = {
            keyword.arg: keyword.value for keyword in matching_calls[0].keywords
        }
        self.assertIsInstance(keywords["default"], ast.Constant)
        self.assertIs(keywords["default"].value, False)

    def test_training_log_uses_existing_timer_and_required_loss(self):
        source = TRAINING_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "log_sft_metric_aliases"
        ]

        self.assertEqual(len(calls), 1)
        keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords
        }
        self.assertEqual(keywords["e2e_step_time_s"], "elapsed_time_per_iteration")
        self.assertEqual(keywords["loss_dict"], "loss_dict")
        self.assertEqual(keywords["enabled"], "args.log_comparison_metrics")
        self.assertEqual(keywords["is_sft"], "args.sft")
        self.assertEqual(keywords["writer"], "wandb_writer")


if __name__ == "__main__":
    unittest.main()

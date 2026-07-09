import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "megatron" / "training" / "sft_metric_aliases.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sft_metric_aliases", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SFTMetricAliasesTest(unittest.TestCase):
    def test_disabled_mode_returns_empty_payload(self):
        module = load_module()

        payload = module.build_sft_metric_aliases(
            enabled=False,
            e2e_step_time_s=51.25,
            main_lm_loss=2.5,
        )

        self.assertEqual(payload, {})

    def test_enabled_mode_preserves_existing_values(self):
        module = load_module()
        loss = object()

        payload = module.build_sft_metric_aliases(
            enabled=True,
            e2e_step_time_s=51.25,
            main_lm_loss=loss,
        )

        self.assertEqual(payload["performance/e2e_step_time_s"], 51.25)
        self.assertIs(payload["accuracy/main_lm_loss"], loss)


if __name__ == "__main__":
    unittest.main()

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

    def test_log_helper_enforces_the_guard_matrix(self):
        module = load_module()

        class Writer:
            def __init__(self):
                self.calls = []

            def log(self, payload, step):
                self.calls.append((payload, step))

        for enabled, is_sft, has_writer in [
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ]:
            writer = Writer() if has_writer else None
            emitted = module.log_sft_metric_aliases(
                writer=writer,
                enabled=enabled,
                is_sft=is_sft,
                iteration=7,
                e2e_step_time_s=51.25,
                loss_dict={"lm loss": object()},
            )
            self.assertFalse(emitted)
            if writer is not None:
                self.assertEqual(writer.calls, [])

    def test_log_helper_emits_one_exact_payload(self):
        module = load_module()

        class Writer:
            def __init__(self):
                self.calls = []

            def log(self, payload, step):
                self.calls.append((payload, step))

        writer = Writer()
        loss = object()
        emitted = module.log_sft_metric_aliases(
            writer=writer,
            enabled=True,
            is_sft=True,
            iteration=7,
            e2e_step_time_s=51.25,
            loss_dict={"lm loss": loss},
        )

        self.assertTrue(emitted)
        self.assertEqual(len(writer.calls), 1)
        payload, step = writer.calls[0]
        self.assertEqual(step, 7)
        self.assertEqual(payload["performance/e2e_step_time_s"], 51.25)
        self.assertIs(payload["accuracy/main_lm_loss"], loss)

    def test_log_helper_fails_when_enabled_sft_loss_is_missing(self):
        module = load_module()

        class Writer:
            def log(self, payload, step):
                raise AssertionError("log must not run without the required loss")

        with self.assertRaises(KeyError):
            module.log_sft_metric_aliases(
                writer=Writer(),
                enabled=True,
                is_sft=True,
                iteration=7,
                e2e_step_time_s=51.25,
                loss_dict={},
            )


if __name__ == "__main__":
    unittest.main()

"""GPU tests for borrowing a fresh CPU snapshot without resizing DDP storage."""

import tempfile
import unittest
import os

import torch
import torch.distributed as dist

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.process_groups_config import ProcessGroupCollection


class TestBorrowedCPUSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        cls.directory = tempfile.TemporaryDirectory()
        dist.init_process_group("nccl", init_method=f"file://{cls.directory.name}/init", rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()
        cls.directory.cleanup()

    def make_buffer(self, grouped: bool = False) -> tuple:
        if grouped:
            from transformer_engine.pytorch.module import GroupedLinear

            model = GroupedLinear(2, 128, 256, bias=False, single_grouped_weight=True,
                                  params_dtype=torch.bfloat16, device="cuda")
            self.assertIsNotNone(getattr(model.weight, "rowwise_data", None))
        else:
            model = torch.nn.Linear(128, 256, bias=False, dtype=torch.bfloat16, device="cuda")
        pairs = list(model.named_parameters())
        group = dist.group.WORLD
        buffer = _ParamAndGradBuffer(
            ddp_config=DistributedDataParallelConfig(use_distributed_optimizer=True),
            param_dtype=torch.bfloat16,
            grad_dtype=torch.bfloat16,
            params_with_names=[(p, n) for n, p in pairs],
            data_parallel_group=group,
            bucket_size=None,
            param_to_name={p: n for n, p in pairs},
            gradient_scaling_factor=1.0,
            param_indices=list(range(len(pairs))),
            nccl_ub=False,
            pg_collection=ProcessGroupCollection(tp=group, dp_cp=group),
        )
        return model, buffer

    def test_freshness_reuse_and_exact_restore(self) -> None:
        for grouped in (False, True):
            with self.subTest(grouped=grouped), torch.no_grad():
                model, buffer = self.make_buffer(grouped)
                buffer.param_data.fill_(2)
                buffer.offload_to_cpu(move_grads=False)
                torch.cuda.synchronize()
                buffer.reload_from_cpu(move_grads=False)
                torch.cuda.synchronize()
                pointer = buffer.param_data.data_ptr()
                cpu_pointer = buffer.param_data_cpu.data_ptr()
                buffer.grad_data.fill_(19)
                for value in (7, 11):
                    buffer.param_data.fill_(value)
                    with buffer.borrow_cpu_param_snapshot() as views:
                        self.assertEqual(buffer.param_data.data_ptr(), pointer)
                        self.assertEqual(buffer.param_data_cpu.data_ptr(), cpu_pointer)
                        buffer.param_data.fill_(23)
                        state = model.state_dict()
                        for name, owner in model.named_parameters():
                            saved = views[owner]
                            self.assertTrue(saved.is_pinned())
                            torch.testing.assert_close(saved, torch.full_like(saved, value), rtol=0, atol=0)
                            state[name].copy_(saved.view(state[name].shape))
                        torch.cuda.synchronize()
                    for owner in model.parameters():
                        live = getattr(owner, "rowwise_data", owner)
                        torch.testing.assert_close(live, torch.full_like(live, value), rtol=0, atol=0)
                    torch.testing.assert_close(buffer.grad_data, torch.full_like(buffer.grad_data, 19), rtol=0, atol=0)

    def test_nested_borrow_and_mutating_lifecycle_rejected(self) -> None:
        _, buffer = self.make_buffer()
        with buffer.borrow_cpu_param_snapshot():
            with self.assertRaises(RuntimeError):
                with buffer.borrow_cpu_param_snapshot():
                    pass
            with self.assertRaises(RuntimeError):
                buffer.offload_to_cpu(move_grads=False)
            with self.assertRaises(RuntimeError):
                buffer.reload_from_cpu(move_grads=False)
        with buffer.borrow_cpu_param_snapshot():
            pass

    def test_release_on_exception(self) -> None:
        _, buffer = self.make_buffer()
        with self.assertRaisesRegex(ValueError, "injected"):
            with buffer.borrow_cpu_param_snapshot():
                raise ValueError("injected")
        with buffer.borrow_cpu_param_snapshot():
            pass


if __name__ == "__main__":
    unittest.main()

import argparse

from megatron.training.arguments import add_megatron_arguments


def test_moe_shared_expert_overlap_can_be_disabled_after_default_flag():
    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)

    assert parser.parse_args([]).moe_shared_expert_overlap is False
    assert parser.parse_args(["--moe-shared-expert-overlap"]).moe_shared_expert_overlap is True
    assert parser.parse_args(
        ["--moe-shared-expert-overlap", "--no-moe-shared-expert-overlap"]
    ).moe_shared_expert_overlap is False

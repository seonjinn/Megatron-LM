# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate"""
import os
import sys
import warnings
import yaml

# Add the directory containing this script to sys.path for local imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Add the megatron-lm root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from argparse import Namespace
from contextlib import nullcontext

from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams

import torch

from model import model_provider as avlm_model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core.enums import ModelType
from megatron.core.inference.engines import AbstractEngine, StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.training import get_model
from megatron.core.transformer.module import MegatronModule
from megatron.inference.text_generation import beam_search_and_post_process
from megatron.inference.text_generation.mcore_engine_server import (
    run_mcore_engine_avlm,
)
from megatron.core.inference.text_generation_controllers.avlm_text_generation_controller import (
    AVLMTextGenerationController,
)
from megatron.core.inference.model_inference_wrappers.multimodal.avlm_inference_wrapper import (
    AVLMInferenceWrapper,
)
from megatron.inference.text_generation_server import MegatronAVLMServer
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore
    backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_seq_length=args.inference_max_seq_length,
        inference_max_requests=args.inference_max_batch_size,
        nccl_all_reduce_for_prefill=args.nccl_all_reduce_for_prefill,
    )

    inference_wrapped_model = AVLMInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = AVLMTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=args.inference_max_batch_size,
    )


def add_text_generate_args(parser):
    """Adds text generation arguments to parser."""
    group = parser.add_argument_group(title='text generation')
    group.add_argument(
        "--port", type=int, default=5000, help='port for text generation server to run on'
    )
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=128,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help='Deprecated in favor of `--inference-max-batch-size`',
    )
    group.add_argument(
        "--out-seq-length", type=int, default=128, help='Length of the output generated text.'
    )
    group.add_argument("--output-path", type=str, help='Output file path')
    group.add_argument('--input-image-path', type=str, help="Input image directory")
    group.add_argument(
        '--num-partitions', type=int, default=0, help="Number of partitions for inputs."
    )
    group.add_argument('--partition-id', type=int, default=0, help="Partition index")
    group.add_argument("--gt-path", type=str, help="Optional ground truth file")
    group.add_argument("--audio-pad-duration", type=float, default=60.0, help="Padding audios to this duration in seconds")
    group.add_argument("--audio-feature-duration", type=float, default=0.08, help="Frame length for encoded audio features in seconds")
    group.add_argument("--audio-sample-rate", type=int, default=16000, help="Sample rate for audio")
    group.add_argument("--audio-start-token", type=str, default="<so_start>", help="Start token for audio")
    group.add_argument("--audio-end-token", type=str, default="<so_end>", help="End token for audio")
    parser = add_multimodal_extra_args(parser)
    return parser


@torch.inference_mode()
def main(model_provider: str = "avlm"):
    """Runs the text generation server with the specified model provider."""
    model_config_yaml = os.environ.get("MODEL_CONFIG", None)
    if model_config_yaml is None and "--model-config" in sys.argv:
        model_config_yaml = sys.argv[sys.argv.index("--model-config") + 1]
        sys.argv.remove("--model-config")
        sys.argv.remove(model_config_yaml)
        print_rank_0(f"Using model config yaml: {model_config_yaml}")

    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'exit_on_missing_checkpoint': True,
        },
        yaml_config=model_config_yaml,
    )
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text " "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    load_context = nullcontext()
    if args.fp8:
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()
    with load_context:
        if model_provider == "avlm":
            def wrapped_model_provider(pre_process, post_process, add_encoder=True, add_decoder=True):
                return avlm_model_provider(pre_process, post_process, add_encoder=add_encoder, add_decoder=add_decoder,
                                    parallel_output=False)
            model = get_model(wrapped_model_provider, model_type=ModelType.encoder_and_decoder, wrap_with_ddp=False)
        else:
            raise ValueError(f"Invalid model provider {model_provider}")

    if args.load is not None:
        _ = load_checkpoint(model, None, None, strict=False)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()

    if args.max_batch_size is not None:
        args.inference_max_batch_size = args.max_batch_size
        warnings.warn(
            "`--max-batch-size` has been deprecated in favor of `--inference-max-requests`, please use `--inference-max-requests` instead"
            f"setting maximum batch size to {args.inference_max_batch_size}"
        )

    inference_engine = get_inference_engine(args, model)

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
            prompts=["Test prompt"], sampling_params=SamplingParams(num_tokens_to_generate=10)
        )

    if (
        mpu.is_pipeline_first_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_expert_model_parallel_rank() == 0
    ):
        server = MegatronAVLMServer(inference_engine, args, mcore_engine_func=run_mcore_engine_avlm, return_generated_text=True)
        server.run("0.0.0.0", port=args.port)

    while True:
        choice = torch.tensor(1, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice.item() == 0:
            try:
                run_mcore_engine_avlm(inference_engine)
            except ValueError as ve:
                pass
        elif choice.item() == 1:
            try:
                beam_search_and_post_process(
                    inference_engine.text_generation_controller.inference_wrapped_model.model
                )
            except ValueError as ve:
                pass


if __name__ == "__main__":
    main(model_provider="avlm")

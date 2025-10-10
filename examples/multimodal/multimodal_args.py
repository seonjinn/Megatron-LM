# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='multimodal arguments')
    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--freeze-sound-model', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=True)
    group.add_argument('--vision-model-type', type=str, default="clip")
    group.add_argument('--sound-model-type', type=str, default=None)
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument(
        "--allow-missing-sound-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument(
        "--allow-missing-sound-model-checkpoint", action="store_true", default=False
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )
    group.add_argument(
        "--use-tiling", action="store_true", default=False, help="Use input image tiling"
    )
    group.add_argument("--max-num-tiles", type=int, default=1, help="Maximum number of image tiles")
    group.add_argument(
        "--use-thumbnail", action="store_true", default=False, help="Add image thumbnail as a tile"
    )
    group.add_argument(
        "--thumbnail-area-threshold", type=float, default=0.8,
        help="Maximum area percentage (0.0-1.0) of resized image relative to thumbnail area for which to add thumbnail. Default 0.8 (80%)"
    )
    group.add_argument(
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
    )
    group.add_argument(
        "--dataloader-seed",
        type=int,
        default=0,
        help="The seed for the dataloader to use for training.",
    )
    group.add_argument(
        "--lr-data-range-start",
        type=int,
        default=0,
        help="Start of the learning rate range as percentage (0-100) of the full training schedule. 0% means start from the beginning of the training schedule. E.g. setting to 10, means start at 10% of the training schedule (the dataloader still starts from the beginning of the dataset, but assume that corresponds to 10% of the training schedule)."
    )
    group.add_argument(
        "--lr-data-range-end",
        type=int,
        default=100,
        help="End of the learning rate range as percentage (0-100) of the full training schedule. 100% means the end of the training schedule. E.g. setting to 90, means end at 90% of the training schedule (the dataloader still ends at the end of the dataset, but assume that corresponds to 90% of the training schedule)."
    )
    group.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to regularly sample from the video as input to the model.",
    )
    group.add_argument(
        "--online-evaluation-config", type=str, help="Config file for online evaluation."
    )
    group.add_argument(
        "--special-tokens",
        nargs="*",
        default=[IMAGE_TOKEN],
        help="Special tokens used in the multimodal model",
    )
    group.add_argument(
        "--tokenizer-prompt-format",
        type=str,
        choices=["mistral", "llama3", "chatml", "nvlm-yi-34b", "qwen2p0", "qwen2p5", "llama3p1", "nemotron5",
                 "nemotron5-aligned", "llama_nemotron_8b", "nemotron-h-reasoning", "nemotron-h-5p5-reasoning",
                 "nemotron-h-5p5-reasoning-inference", "llama-nemotron-super", "llama-nemotron-super-1p5"],
        required=True,
        help="Prompt format to use with the tokenizer.",
    )
    group.add_argument("--pixel-shuffle", action="store_true", default=False)
    group.add_argument(
        "--image-tag-type",
        type=str,
        choices=["nvlm", "internvl", ""],
        default="",  # Default: Image tag not used.
        help="Surround image tokens with tags.",
    )
    group.add_argument("--use-tile-tags", action="store_true", default=False, help="Use tile tags")
    group.add_argument(
        "--packing-buffer-size",
        type=int,
        default=None,   # Packing is disabled by default.
        help="Enable sample packing by setting the buffer size to > 0",
    )
    group.add_argument(
        "--packing-seq-length", type=int, default=0, help="Packing sequence length. Must be > 0 if using packing."
    )
    group.add_argument(
        "--packing-knapsack-algorithm", type=str, default="greedy_knapsack", help="Knapsack algorithm to use for packing."
    )
    group.add_argument(
        "--recompute-vision", action="store_true", default=False, help="Enable activation checkpointing in the vision model"
    )
    group.add_argument(
        "--recompute-sound", action="store_true", default=False, help="Enable activation checkpointing in the sound model"
    )
    group.add_argument(
        "--use-loss-scaling", action="store_true", default=False, help="Scale loss based on conversation turn length (in tokens)."
    )
    group.add_argument(
        "--force-system-message", action="store_true", default=False, help="Force a specific system message"
    )
    group.add_argument("--eos-id", type=int, help="termination id for MultiModal Tokenizer")
    group.add_argument(
        "--use-area-weighted-aspect-ratio", action="store_true", default=False,
        help=(
            "When --use-tiling is True, find the aspect ratio to use based on the original ",
            "image aspect ratio and the area covered by the tiles.")
    )
    group.add_argument("--use-mcore-inference", action="store_true", default=False, help="Use the MCore inference API")
    group.add_argument("--use-vision-backbone-fp8-arch", action="store_true", default=False, help="Use the FP8 arch in the vision backbone. This is used to load the FP8 checkpoint when running inference.")
    group.add_argument(
        "--dynamic-resolution", action="store_true", default=False, help="Use input image dynamic resolution"
    )
    group.add_argument(
        "--dynamic-resolution-min-patches", type=int, default=0, help="Minimum number of patches per image for dynamic resolution"
    )
    group.add_argument(
        "--dynamic-resolution-min-side", type=int, default=None, help="Minimum side length for dynamic resolution"
    )
    group.add_argument(
        "--match-tiling-dynamic-resolution", action="store_true", default=False,
        help="Use match-tiling dynamic resolution strategy that combines tiling logic with dynamic resolution processing"
    )
    group.add_argument(
        "--image-break-token", type=str, default=None, help="Token to use for image break tokens, must be added to --special-tokens as well"
    )
    group.add_argument("--conv-merging", action="store_true", default=False, help="Use convolution merging which uses a convolution to merge tokens after the vision encoder")
    group.add_argument(
        "--allow-missing-conv-merge-checkpoint", action="store_true", default=False
    )
    group.add_argument(
        "--video-min-num-frames", type=int, default=8, help="Minimum number of frames to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-max-num-frames", type=int, default=32, help="Maximum number of frames to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-default-fps", type=int, default=2, help="Default frames per second to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-frame-temporal-jitter", action="store_true", default=False, help="Enable temporal jittering of the frames to sample from the video as input to the model.",
    )
    group.add_argument(
        "--enable-fusions", action="store_true", default=True, help="Enable fusions in the model."
    )
    group.add_argument(
        "--optimize-broadcast", action="store_true", default=True, help="Optimize the broadcast of data.",
    )
    group.add_argument(
        "--recompute-vision-num-layers", type=int, default=0, help="Number of layers to recompute in the vision model."
    )
    group.add_argument(
        "--recompute-granularity-vision", type=str, default=None, help="Granularity to recompute in the vision model.",
        choices=["full", "selective"],
    )
    group.add_argument(
        "--recompute-method-vision", type=str, default=None,
        choices=['uniform', 'block'], help="Method to recompute in the vision model.",
    )
    group.add_argument(
        "--allow-large-videos", action="store_true", default=False, help="Allow large videos to be loaded into the model."
    )
    group.add_argument(
        "--efficient-video-sampling-variant", type=str, default=None, help="The EVS variant. Read docstring on EVSHelper"
    )
    group.add_argument(
        "--sound-target-rate",
        type=int,
        default=16000,
        help="Target rate of sound clips to regularly sample from the audio as input to the model.",
    )
    group.add_argument(
        "--sound-embedding-size",
        type=int,
        default=750,
        help="Size of the sound embedding.",
    )
    group.add_argument(
        "--sound-clip-duration",
        type=int,
        default=30,
        help="Sound model clip duration in seconds."
    )


    return parser

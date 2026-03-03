Steps to train an omni model with Conv3D temporal compression:

1. Grab a trained VLM + text mode, either 16k or 49k, whichever is best.
2. Train the A1 stage using the pretrain_moe_a1.sh script. Once trained, select the checkpoint with the lowest ASR error.
3. Train the A2 stage using the pretrain_moe_a2_full_blend.sh script. Once trained, select the checkpoint with the lowest ASR error and highest MMAU average score.
4. Run the 16k SFT stage using the sft_13p70.sh script.
4. Run the long context SFT stage using the sft_long_context.sh script.

All scripts are compatible with examples/multimodal/launch.sh. Example:

    examples/multimodal/launch.sh \
      --name pretrain_moe_a1_conv3d_0220 \
      --sbatch examples/multimodal/v3_omni_staged_conv3d/pretrain_moe_a1.sh \
      --num-jobs 5

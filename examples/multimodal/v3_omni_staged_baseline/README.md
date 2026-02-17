Steps to train an omni model:

1. Grab a trained VLM + text mode, either 16k or 49k, whichever is best.
2. Train the A1 stage using the pretrain_moe_a1.sh script. Once trained, select the checkpoint with the lowest ASR error.
3. Train the A2 stage using the pretrain_moe_a2_full_blend.sh script. Once trained, select the checkpoint with the lowest ASR error and highest MMAU average score.
4. Run the 16k SFT stage using the sft_lower_lr_13p67_from_vlm.sh script.

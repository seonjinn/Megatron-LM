# Tools Guide

We put in this folder a collection of tools / scripts we use to prepare the initial checkpoint used for training.

Note that these scripts are not polished, but they're also simple enough: make sure to read them before you run them to update the paths, tensor parallelism, or other hard coded options.

## Pre-requisite

The scripts require models to be in the torch format. The LLM team often provides models in the distributed checkpoint format. To convert, we ask the model owner for the training script, docker sqsh file, and megatron-lm branch used to train the model. Then, we update the training script with:

```
    --load ${PATH_TO_MODEL_TO_LOAD} \
    --ckpt-convert-format torch \
    --ckpt-convert-save ${OUTPUT_PATH}
```

We then run the training script in an interactive session using the provided sqsh file and these new options. It will load the model, save the torch converted checkpoint to OUTPUT_PATH, and exit.

## Prepare LLM

The `prepare_llm.py` script loads the LLM and does two things:
1. It re-initializes the first 1000 tokens input and output embeddings to the average / standard deviation of the rest of the embeddings. This is because nemotron pretraining uses 1000 special token embeddings which are not used during training and converge to 0 because of weight decay.
2. It pads the end of the embeddings to either 132096 or 131584 depending on the tensor parallelism so that the embeddings have a shape leading to more efficient GEMM.

## Replace LLM backbone

The `replace_llm_backbone.py` script loads an already combined LLM / Vision backbone combination and replaces the LLM with a newly provided one. This script is used as a convenience to create a LLM / Vision combination leveraging the ones existing in `/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/`.
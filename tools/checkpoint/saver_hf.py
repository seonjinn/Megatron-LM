import os
import sys

import torch

from schema_hf import get_language_model_schema

from huggingface_hub import save_torch_state_dict

def add_arguments(parser):
    group = parser.add_argument_group(title='HuggingFace Language Model saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

class HFCheckpointSaver:
    def __init__(self, args, queue):
        self.args = args
        self.queue = queue

    def insert_megatron_path(self):
        sys.path.append(os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         os.path.pardir,
                         os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

    def queue_get(self, name=None):
        val = self.queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and self.args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(self, msg):
        if not self.args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    def recover_lm_qkv_weight(self, qkv_weight):
        dim = self.md.hidden_size // self.md.num_attention_heads
        tp = self.md.previous_tensor_parallel_size
        nh = self.md.num_attention_heads // tp
        ng = self.md.num_query_groups // tp
        hidden_size = self.md.hidden_size

        params_per_tp = torch.chunk(qkv_weight, tp, dim=0)

        # Use lists to collect tensors, then concatenate once at the end
        # This preserves dtype and is more efficient
        q_parts = []
        k_parts = []
        v_parts = []

        for t in params_per_tp:
            # 1. Reshape back to (ng, (dim*nh//ng + 2*dim), hidden_size).
            qkv = t.reshape(ng, dim * (nh // ng) + 2 * dim, -1)

            # 2. Slice out q, k, v along dim=1.
            q_t = qkv[:, : dim * (nh // ng), :]  
            k_t = qkv[:, dim * (nh // ng) : dim * (nh // ng) + dim, :]
            v_t = qkv[:, dim * (nh // ng) + dim :, :]

            # 3. Reshape each to match the original HF shapes.
            q_t = q_t.reshape(ng, dim * (nh // ng), -1)
            k_t = k_t.reshape(ng, dim, -1)
            v_t = v_t.reshape(ng, dim, -1)

            qp = q_t.reshape(dim * (nh // ng) * ng, -1)
            kp = k_t.reshape(dim * ng, -1)
            vp = v_t.reshape(dim * ng, -1)

            q_parts.append(qp)
            k_parts.append(kp)
            v_parts.append(vp)

        # Concatenate all parts at once - this preserves the original dtype
        q = torch.cat(q_parts, dim=0)
        k = torch.cat(k_parts, dim=0)  
        v = torch.cat(v_parts, dim=0)

        return q, k, v

    def recover_lm_qkv_bias(self, qkv_bias):
        dim = self.md.hidden_size // self.md.num_attention_heads
        tp = self.md.previous_tensor_parallel_size
        nh = self.md.num_attention_heads // tp
        ng = self.md.num_query_groups // tp

        bias_per_tp = torch.chunk(qkv_bias, tp, dim=0)

        # Use lists to collect tensors, then concatenate once at the end
        # This preserves dtype and is more efficient
        qb_parts = []
        kb_parts = []
        vb_parts = []

        for b in bias_per_tp:
            qkvb = b.reshape(ng, dim * (nh // ng) + 2 * dim)

            q_b = qkvb[:, : dim * (nh // ng)]  
            k_b = qkvb[:, dim * (nh // ng) : dim * (nh // ng) + dim]
            v_b = qkvb[:, dim * (nh // ng) + dim :]

            q_b = q_b.reshape(ng, dim * (nh // ng))
            k_b = k_b.reshape(ng, dim)
            v_b = v_b.reshape(ng, dim)

            q_b = q_b.reshape(-1)
            k_b = k_b.reshape(-1)
            v_b = v_b.reshape(-1)

            qb_parts.append(q_b) 
            kb_parts.append(k_b) 
            vb_parts.append(v_b) 

        # Concatenate all parts at once - this preserves the original dtype
        qb = torch.cat(qb_parts, dim=0)
        kb = torch.cat(kb_parts, dim=0)
        vb = torch.cat(vb_parts, dim=0)

        return qb, kb, vb

    def receive_lm(self, schema):
        embeddings_msg = self.queue_get("embeddings")
        params_dict = {}

        params_dict["word_embeddings"] = embeddings_msg["word embeddings"]
        if self.md.position_embedding_type == "learned_absolute":
            # TODO: Below may not be correct, but none of our models use absolute positional embeddings
            params_dict["position_embeddings"] = embeddings_msg["position embeddings"]

        schema.set(self.state_dict, params_dict)

        #TODO: maybe refactor hybrid layer code into separate functions
        if self.md.model_type == "hybrid":
            from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
            from megatron.core.ssm.mamba_hybrid_layer_allocation import allocate_layers

            #TODO: maybe refactor these layer things into separate functions
            layer_type_list = allocate_layers(
                self.md.num_layers,
                self.md.hybrid_attention_ratio,
                self.md.hybrid_mlp_ratio,
                self.md.hybrid_override_pattern,
            )

            for i in range(self.md.num_layers):
                message = self.queue_get(f"transformer layer {i}")
                params_dict = {}

                layer_type = layer_type_list[i]
                if layer_type == LayerSymbols.MAMBA:
                    # looks weird but this might be it
                    params_dict["norm_weight"] = message["in proj norm weight"]
                    params_dict["mixer_norm_weight"] = message["norm weight"]
                    params_dict["mixer_D"] = message["D"]
                    params_dict["mixer_dt_bias"] = message["dt bias"]
                    params_dict["mixer_A_log"] = message["A log"]
                    params_dict["mixer_in_proj_weight"] = message["in proj weight"]
                    params_dict["mixer_conv1d_weight"] = message["conv1d weight"]
                    params_dict["mixer_conv1d_bias"] = message["conv1d bias"]
                    params_dict["mixer_out_proj_weight"] = message["out proj weight"]
                elif layer_type == LayerSymbols.ATTENTION:
                    params_dict["norm_weight"] = message["input norm weight"]
                    if self.md.norm_has_bias:
                        params_dict["norm bias"] = message["input norm bias"]

                    qkv_weight = message["qkv weight"]
                    if self.md.qkv_bias:
                        qkv_bias = message["qkv bias"]

                    q, k, v = self.recover_lm_qkv_weight(qkv_weight)
                    q = q.clone().detach().contiguous()
                    params_dict["q_proj_weight"] = q
                    k = k.clone().detach().contiguous()
                    params_dict["k_proj_weight"] = k
                    v = v.clone().detach().contiguous()
                    params_dict["v_proj_weight"] = v

                    if self.md.qkv_bias:
                        qb, kb, vb = self.recover_lm_qkv_bias(qkv_bias)
                        qb = qb.clone().detach().contiguous()
                        params_dict["q_proj_bias"] = qb
                        kb = kb.clone().detach().contiguous()
                        params_dict["k_proj_bias"] = kb
                        vb = vb.clone().detach().contiguous()
                        params_dict["v_proj_bias"] = vb

                    params_dict["dense_weight"] = message["dense weight"]
                    if self.md.linear_bias:
                        params_dict["dense_bias"] = message["dense bias"]
                elif layer_type == LayerSymbols.MLP:
                    params_dict["norm_weight"] = message["post norm weight"]
                    if self.md.norm_has_bias:
                        params_dict["norm bias"] = message["post norm bias"]

                    params_dict["mlp_l0_weight"] = message["mlp l0 weight"]
                    if self.md.linear_bias:
                        params_dict["mlp_l0_bias"] = message["mlp l0 bias"] 

                    params_dict["mlp_l1_weight"] = message["mlp l1 weight"]
                    if self.md.linear_bias:
                        params_dict["mlp_l1_bias"] = message["mlp l1 bias"]
                else:
                    raise ValueError(f"hybrid layer {i} is not one of MAMBA, ATTENTION, or MLP")

                schema.set_layer(self.state_dict, i, params_dict)
        else:
            for i in range(self.md.num_layers):
                message = self.queue_get(f"transformer layer {i}")
                params_dict = {}

                params_dict["input_norm_weight"] = message["input norm weight"]
                params_dict["post_norm_weight"] = message["post norm weight"]
                if self.md.norm_has_bias:
                    params_dict["input_norm_bias"] = message["input norm bias"]
                    params_dict["post_norm_bias"] = message["post norm bias"]

                if self.md.swiglu:
                    params_dict["mlp_l0_weight_W"] = message["mlp l0 weight W"]
                    params_dict["mlp_l0_weight_V"] = message["mlp l0 weight V"]
                else:
                    params_dict["mlp_l0_weight"] = message["mlp l0 weight"]
                if self.md.linear_bias:
                    if self.md.swiglu:
                        params_dict["mlp_l0_bias_W"] = message["mlp l0 bias W"]
                        params_dict["mlp_l0_bias_V"] = message["mlp l0 bias V"]
                    else:
                        params_dict["mlp_l0_bias"] = message["mlp l0 bias"] 

                qkv_weight = message["qkv weight"]
                if self.md.qkv_bias:
                    qkv_bias = message["qkv bias"]

                q, k, v = self.recover_lm_qkv_weight(qkv_weight)
                q = q.clone().detach().contiguous()
                params_dict["q_proj_weight"] = q
                k = k.clone().detach().contiguous()
                params_dict["k_proj_weight"] = k
                v = v.clone().detach().contiguous()
                params_dict["v_proj_weight"] = v

                if self.md.qkv_bias:
                    qb, kb, vb = self.recover_lm_qkv_bias(qkv_bias)
                    qb = qb.clone().detach().contiguous()
                    params_dict["q_proj_bias"] = qb
                    kb = kb.clone().detach().contiguous()
                    params_dict["k_proj_bias"] = kb
                    vb = vb.clone().detach().contiguous()
                    params_dict["v_proj_bias"] = vb

                params_dict["mlp_l1_weight"] = message["mlp l1 weight"]
                params_dict["dense_weight"] = message["dense weight"]
                if self.md.linear_bias:
                    params_dict["mlp_l1_bias"] = message["mlp l1 bias"]
                    params_dict["dense_bias"] = message["dense bias"]

                schema.set_layer(self.state_dict, i, params_dict)

        params_dict = {
            "final_norm": self.queue_get('final norm')['weight'],
            "output_layer": self.queue_get('output layer')['weight'],

        }
        schema.set(self.state_dict, params_dict)

        msg = self.queue_get()
        if msg != "done":
            print("ERROR: got some more data but was expecting to be done")

    def receive_model(self):
        language_model_prefix = ""
        if self.md.model_type == "hybrid":
            language_layer_prefix = "backbone.layers"
        else:
            language_layer_prefix = "model.layers"
        language_schema = get_language_model_schema(
            model_type=self.args.model_type,
            prefix=language_model_prefix,
            layer_prefix=language_layer_prefix,
            use_swiglu=self.md.swiglu,
        )
        self.receive_lm(language_schema)

    def save_state_dict_to_hf_checkpoint(self):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # Store the state_dict to file.
        max_shard_size = "4GB"
        save_torch_state_dict(self.state_dict, self.args.save_dir, max_shard_size=max_shard_size)

    def save(self):
        self.insert_megatron_path()

        self.md = self.queue_get()
        
        self.state_dict = {}

        self.receive_model()

        self.save_state_dict_to_hf_checkpoint()

        print("Done!")
        

def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().
    """
    saver = HFCheckpointSaver(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e 

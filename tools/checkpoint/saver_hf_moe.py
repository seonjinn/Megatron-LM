import os
import sys

import torch

from schema_hf import get_language_model_schema
from saver_hf import HFCheckpointSaver

from huggingface_hub import save_torch_state_dict


def add_arguments(parser):
    group = parser.add_argument_group(title='HuggingFace MoE Model saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')


class HFCheckpointSaverMoE(HFCheckpointSaver):
    def __init__(self, args, queue):
        super().__init__(args, queue)

    def _receive_moe_layer(self, message: dict) -> dict:
        """
        Convert loader MoE message into a params_dict consumable by the MoE-aware HF schema.
        Returns keys that HFHybridMoELMSchema understands, including stacked expert weights.
        """
        params_dict = {}

        # Norm: for hybrid schema we use "norm_weight"/"norm_bias" (already mapped by schema)
        params_dict["norm_weight"] = message["pre mlp norm weight"]
        if self.md.norm_has_bias:
            params_dict["norm_bias"] = message["pre mlp norm bias"]

        # Router and shared experts
        params_dict["router_weight"] = message["router weight"]
        params_dict["router_bias"] = message["router bias"].to(torch.float32)
        params_dict["shared_up_proj_weight"] = message["shared mlp l0 weight"]
        params_dict["shared_down_proj_weight"] = message["shared mlp l1 weight"]

        # Experts (stacked on dim 0)
        #TODO: maybe handle swiglu case?
        experts_up = message["mlp l0 weight"]  # [E, out, in]
        params_dict["experts_up_proj_weight"] = experts_up
        params_dict["experts_down_proj_weight"] = message["mlp l1 weight"]

        # MoE latent projections (replicated across TP, not sharded)
        moe_latent_size = getattr(self.md, 'moe_latent_size', None)
        if moe_latent_size:
            params_dict["fc1_latent_proj_weight"] = message["fc1 latent proj weight"]
            params_dict["fc2_latent_proj_weight"] = message["fc2 latent proj weight"]

        return params_dict

    def _write_mtp_layer(self, msg, mtp_layer_idx):
        """Write MTP layer weights to HF state dict.

        Handles hybrid MTP patterns (e.g. '*E'): outer params (enorm/hnorm/eh_proj/
        final_layernorm) are written to the first/last logical HF layer; sub-layer
        params (attention, MoE) are written to their respective HF layers.

        The loader sends sub-layer params when the hybrid MTP override pattern is set
        (see loader_base.py::_send_mtp_hybrid_sublayers).  All pops use a None default
        so missing keys are silently skipped.
        """
        mtp_pattern = getattr(self.md, 'mtp_hybrid_override_pattern', None)
        if not mtp_pattern:
            # Non-hybrid MTP: not yet implemented for HF conversion; drain silently.
            return

        from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols

        layer_type_list = [c for c in mtp_pattern if c != LayerSymbols.PIPE]
        pattern_len = len(layer_type_list)
        hf_base = mtp_layer_idx * pattern_len

        # --- Outer MTP params ---
        enorm_w = msg.pop("enorm weight", None)
        hnorm_w = msg.pop("hnorm weight", None)
        eh_proj_w = msg.pop("eh proj weight", None)
        final_norm_w = msg.pop("final layernorm weight", None)

        if enorm_w is not None:
            self.state_dict[f"mtp.layers.{hf_base}.enorm.weight"] = enorm_w.clone()
        if hnorm_w is not None:
            self.state_dict[f"mtp.layers.{hf_base}.hnorm.weight"] = hnorm_w.clone()
        if eh_proj_w is not None:
            self.state_dict[f"mtp.layers.{hf_base}.eh_proj.weight"] = eh_proj_w.clone()
        if final_norm_w is not None:
            last_hf = hf_base + pattern_len - 1
            self.state_dict[f"mtp.layers.{last_hf}.final_layernorm.weight"] = final_norm_w.clone()

        # --- Sub-layer params (present when loader sends hybrid sublayers) ---
        for sub_idx, layer_type in enumerate(layer_type_list):
            hf_idx = hf_base + sub_idx

            if layer_type == LayerSymbols.ATTENTION:
                input_norm = msg.pop("input norm weight", None)
                if input_norm is not None:
                    # vLLM's NemotronHMTPModel reads `mtp.layers.{i}.norm.weight` and
                    # remaps to `model.layers.{i}.norm.weight`, matching the convention
                    # used by vLLM's main NemotronH decoder layers.
                    self.state_dict[f"mtp.layers.{hf_idx}.norm.weight"] = input_norm.clone()

                qkv = msg.pop("qkv weight", None)
                if qkv is not None:
                    q, k, v = self.recover_lm_qkv_weight(qkv)
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.q_proj.weight"] = q.clone().contiguous()
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.k_proj.weight"] = k.clone().contiguous()
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.v_proj.weight"] = v.clone().contiguous()

                dense = msg.pop("dense weight", None)
                if dense is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.o_proj.weight"] = dense.clone()

            elif layer_type == LayerSymbols.MOE:
                pre_norm = msg.pop("pre mlp norm weight", None)
                if pre_norm is not None:
                    # See note above: vLLM expects `norm.weight`, not `input_layernorm.weight`.
                    self.state_dict[f"mtp.layers.{hf_idx}.norm.weight"] = pre_norm.clone()

                router_w = msg.pop("router weight", None)
                if router_w is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.gate.weight"] = router_w.clone()
                router_b = msg.pop("router bias", None)
                if router_b is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.gate.e_score_correction_bias"] = router_b.to(torch.float32).clone()

                fc1_lat = msg.pop("fc1 latent proj weight", None)
                if fc1_lat is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.fc1_latent_proj.weight"] = fc1_lat.clone()
                fc2_lat = msg.pop("fc2 latent proj weight", None)
                if fc2_lat is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.fc2_latent_proj.weight"] = fc2_lat.clone()

                experts_up = msg.pop("mlp l0 weight", None)  # [E, out, in]
                if experts_up is not None:
                    for e in range(experts_up.shape[0]):
                        self.state_dict[f"mtp.layers.{hf_idx}.mixer.experts.{e}.up_proj.weight"] = experts_up[e].clone()

                experts_down = msg.pop("mlp l1 weight", None)  # [E, out, in]
                if experts_down is not None:
                    for e in range(experts_down.shape[0]):
                        self.state_dict[f"mtp.layers.{hf_idx}.mixer.experts.{e}.down_proj.weight"] = experts_down[e].clone()

                shared_up = msg.pop("shared mlp l0 weight", None)
                if shared_up is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.shared_experts.up_proj.weight"] = shared_up.clone()
                shared_down = msg.pop("shared mlp l1 weight", None)
                if shared_down is not None:
                    self.state_dict[f"mtp.layers.{hf_idx}.mixer.shared_experts.down_proj.weight"] = shared_down.clone()

    def receive_lm(self, schema):
        # Embeddings
        embeddings_msg = self.queue_get("embeddings")
        params_dict = {}

        params_dict["word_embeddings"] = embeddings_msg["word embeddings"]
        if self.md.position_embedding_type == "learned_absolute":
            params_dict["position_embeddings"] = embeddings_msg["position embeddings"]
        schema.set(self.state_dict, params_dict)

        # Hybrid path: allocate layers and branch
        if self.md.model_type == "hybrid":
            from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols

            # The saver has already merged all PP stages, so read layer types directly
            # from the pattern rather than calling allocate_layers(), which requires
            # torch.distributed to be initialized.
            layer_type_list = [
                c for c in self.md.hybrid_override_pattern if c != LayerSymbols.PIPE
            ]

            for i in range(self.md.num_layers):
                message = self.queue_get(f"transformer layer {i}")

                layer_type = layer_type_list[i]
                if layer_type == LayerSymbols.MAMBA:
                    params_dict = self._receive_mamba_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                elif layer_type == LayerSymbols.ATTENTION:
                    params_dict = self._receive_attention_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                elif layer_type == LayerSymbols.MLP:
                    params_dict = self._receive_mlp_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                elif layer_type == LayerSymbols.MOE:
                    params_dict = self._receive_moe_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                else:
                    raise ValueError(f"hybrid layer {i} is not one of MAMBA, ATTENTION, MLP, or MOE")
        else:
            raise ValueError("Non-hybrid model is not supported for MoE")

        # MTP layers (present when mtp_num_layers > 0)
        mtp_num_layers = getattr(self.md, 'mtp_num_layers', None)
        if mtp_num_layers is not None and mtp_num_layers > 0:
            mtp_use_repeated = getattr(self.md, 'mtp_use_repeated_layer', False)
            num_physical = 1 if mtp_use_repeated else mtp_num_layers
            for i in range(num_physical):
                msg = self.queue_get(f"mtp layer {i}")
                self._write_mtp_layer(msg, i)

        # Final norms and output layer
        params_dict = {
            "final_norm": self.queue_get('final norm')['weight'],
            "output_layer": self.queue_get('output layer')['weight'],
        }
        schema.set(self.state_dict, params_dict)

        # MTP draft-model duplicates: vLLM's NemotronHMTPModel.load_weights()
        # filters by `name.startswith("mtp.")` or `"embeddings"/"lm_head" in name`,
        # then renames `embeddings -> embed_tokens` and (if name starts with
        # `backbone.`) `backbone. -> model.`. Our LLaVA HF schema writes the
        # tensors under `language_model.backbone.embeddings.weight` and
        # `language_model.lm_head.weight`, neither of which the MTP loader can
        # match. Add unprefixed aliases so the MTP draft can find them.
        # We must `.clone()` because safetensors deduplicates shared-storage
        # tensors and keeps only the lexically-first name, which would drop
        # the prefixed keys that the main VLM model loader needs.
        if mtp_num_layers is not None and mtp_num_layers > 0:
            embed_keys = [k for k in self.state_dict
                          if k.endswith("backbone.embeddings.weight")
                          and k != "backbone.embeddings.weight"]
            lm_head_keys = [k for k in self.state_dict
                            if k.endswith("lm_head.weight")
                            and k != "lm_head.weight"]
            if embed_keys:
                self.state_dict["backbone.embeddings.weight"] = self.state_dict[embed_keys[0]].clone()
            if lm_head_keys:
                self.state_dict["lm_head.weight"] = self.state_dict[lm_head_keys[0]].clone()

        msg = self.queue_get()
        if msg != "done":
            print("ERROR: got some more data but was expecting to be done")


def save_checkpoint(queue, args):
    """
    Entry point for LM-only MoE and hybrid models.
    """
    saver = HFCheckpointSaverMoE(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e

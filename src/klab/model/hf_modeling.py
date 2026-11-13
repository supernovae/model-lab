"""Hugging Face ``transformers`` wrapper around Lab's decoder.

``klab weights export --format hf_remote_code`` copies this module into the
export, and ``config.json`` names its classes in ``auto_map``, so

    AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True)

builds :class:`DenseLM` from the same modules Lab trains with. The wrapper
adds no model math. ``generate`` decodes with Lab's own incremental cache
(``DenseLM.forward_cached``) and follows ``klab generate``'s context policy:
the newest ``max_position_embeddings`` tokens are read, and a full cache is
rebuilt from them. Greedy decoding therefore gives ``klab generate
--temperature 0``'s tokens. Padded batches are refused.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from klab.model.export_loader import build_model


class KlabConfig(PretrainedConfig):
    model_type = "klab"

    def __init__(
        self, klab_architecture: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        self.klab_architecture = klab_architecture
        super().__init__(**kwargs)


def _unpadded(attention_mask: torch.Tensor | None) -> None:
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("Lab models do not read padded batches; pass one row each")


class KlabForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = KlabConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    # The decoder keeps its own cache object instead of a transformers Cache.
    _is_stateful = True

    def __init__(self, config: KlabConfig) -> None:
        super().__init__(config)
        self.model = build_model(config.klab_architecture or {})
        if config.tie_word_embeddings:
            self._tied_weights_keys = ["model.output.weight"]
        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Weights come from the export; the decoder initializes itself."""

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embedding

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        _unpadded(attention_mask)
        cache = None
        if past_key_values is not None:
            logits, cache = self.model.forward_cached(input_ids, cache=past_key_values)
        elif use_cache:
            logits, cache = self.model.forward_cached(
                input_ids, cache_capacity=self.config.max_position_embeddings
            )
        else:
            logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits[:, :-1].flatten(0, 1).float(),
                labels[:, 1:].flatten(),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _unpadded(attention_mask)
        context = input_ids[:, -self.config.max_position_embeddings :]
        if not (use_cache and self.model.incremental_cache_capability.supported):
            return {"input_ids": context, "use_cache": False}
        if past_key_values is not None and past_key_values.length < (
            past_key_values.capacity
        ):
            return {
                "input_ids": input_ids[:, -1:],
                "past_key_values": past_key_values,
                "use_cache": True,
            }
        # A new request, or a full context: start a cache from the newest tokens.
        return {"input_ids": context, "use_cache": True}

import torch
import torch.nn as nn
from transformers import LlamaPreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    logger,
    _CONFIG_FOR_DOC,
    LLAMA_INPUTS_DOCSTRING
)
from transformers.utils import add_start_docstrings_to_model_forward, logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple, Union

class LlamaModelWithLayerLoop(LlamaModel):
    """
    LlamaModel subclass that allows looping over specified transformer layers multiple times during the forward pass.

    Args:
        config (LlamaConfig): Model configuration.
        loop_start (int, optional): The starting layer index for looping. Defaults to 0.
        loop_end (int, optional): The ending layer index for looping (exclusive). Defaults to None.
        loop_times (int, optional): Number of times to loop over the specified layers. Defaults to 1.
    """

    def __init__(self, config: LlamaConfig, loop_start: int = 0, loop_end: Optional[int] = None, loop_times: int = 1):
        super().__init__(config)
        self.loop_start = loop_start
        self.loop_end = loop_end if loop_end is not None else config.num_hidden_layers
        self.loop_times = loop_times

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Retrieve loop parameters from kwargs if provided, else use class attributes
        loop_start = kwargs.get('loop_start', self.loop_start)
        loop_end = kwargs.get('loop_end', self.loop_end)
        loop_times = kwargs.get('loop_times', self.loop_times)

        # Validate loop parameters
        if loop_start < 0 or loop_start >= len(self.layers):
            raise ValueError(f"Invalid loop_start value: {loop_start}. It must be in [0, {len(self.layers) - 1}]")
        if loop_end <= loop_start or loop_end > len(self.layers):
            raise ValueError(f"Invalid loop_end value: {loop_end}. It must be in ({loop_start}, {len(self.layers)}]")
        if loop_times < 1:
            raise ValueError(f"Invalid loop_times value: {loop_times}. It must be >= 1")

        # Standard forward method code
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Input validation
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # Prepare inputs
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = inputs_embeds.shape[:-1]

        # Handle past_key_values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        else:
            if not len(past_key_values) == len(self.layers):
                raise ValueError(
                    f"past_key_values length ({len(past_key_values)}) doesn't match the number of layers ({len(self.layers)})"
                )

        # Input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_len), hidden_states, past_key_values_length=0)

        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Process layers before loop_start
        for idx in range(loop_start):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_module = self.layers[idx]
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1 if output_attentions else 0],)

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        # Loop over specified layers
        for _ in range(loop_times):
            for idx in range(loop_start, loop_end):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_module = self.layers[idx]
                past_key_value = past_key_values[idx] if past_key_values is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[1 if output_attentions else 0],)

                if output_attentions:
                    all_self_attentions += (layer_outputs[1],)

        # Process layers after loop_end
        for idx in range(loop_end, len(self.layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_module = self.layers[idx]
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1 if output_attentions else 0],)

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Prepare outputs
        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_decoder_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attentions,)
            return outputs  # type: ignore

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # Prepare inputs for generation, passing loop parameters
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        inputs['loop_start'] = kwargs.get('loop_start', self.loop_start)
        inputs['loop_end'] = kwargs.get('loop_end', self.loop_end)
        inputs['loop_times'] = kwargs.get('loop_times', self.loop_times)
        return inputs
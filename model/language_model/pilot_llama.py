#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..pilot_arch import pilotMetaModel, pilotMetaForCausalLM


class pilotConfig(LlamaConfig):
    model_type = "pilot"


class pilotLlamaModel(pilotMetaModel, LlamaModel):
    config_class = pilotConfig

    def __init__(self, config: LlamaConfig):
        super(pilotLlamaModel, self).__init__(config)


class pilotLlamaForCausalLM(LlamaForCausalLM, pilotMetaForCausalLM):
    config_class = pilotConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = pilotLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                Diff_loss,
                mlp_balance_loss,
                mlp_router_z_loss
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
            # -------------------------------
        output_router_logits = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            # output_router_logits = output_router_logits,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            Diff_loss = Diff_loss.sum(dim=-1).mean()
            Diff_loss = self.config.Diff_loss_coef * Diff_loss
            loss += Diff_loss

        if self.config.training:
            if self.config.mlp_smoe: #  or self.config.clip_smoe
                if self.config.local_rank == 0:
                    print('language loss: ', loss.item())
                if self.config.mlp_smoe:
                    mlp_balance_loss = mlp_balance_loss.sum(dim=-1).mean()
                    mlp_balance_loss = self.config.balance_loss_coef * mlp_balance_loss
                    loss += mlp_balance_loss
                    mlp_router_z_loss = mlp_router_z_loss.sum(dim=-1).mean()
                    mlp_router_z_loss = self.config.router_z_loss_coef * mlp_router_z_loss
                    loss += mlp_router_z_loss
                    Diff_loss = Diff_loss.sum(dim=-1).mean()
                    Diff_loss = self.config.Diff_loss_coef * Diff_loss
                    loss += Diff_loss
                    if self.config.local_rank == 0:
                        print('mlp balance loss: ', mlp_balance_loss.item(), 'mlp router z loss: ',
                              mlp_router_z_loss.item(), 'Diff_loss', Diff_loss.item())
                # if self.config.clip_smoe:
                #     clip_balance_loss = clip_balance_loss.sum(dim=-1).mean()
                #     clip_balance_loss = self.config.balance_loss_coef * clip_balance_loss
                #     loss += clip_balance_loss
                #     clip_router_z_loss = clip_router_z_loss.sum(dim=-1).mean()
                #     clip_router_z_loss = self.config.router_z_loss_coef * clip_router_z_loss
                #     loss += clip_router_z_loss
                #     if self.config.local_rank == 0:
                #         print('clip balance loss: ', clip_balance_loss.item(), 'clip router z loss: ',
                #               clip_router_z_loss.item())

                # balance_loss = [loss_pair[0] for loss_pair in outputs.router_logits]
                # b_loss = sum(balance_loss) / len(balance_loss)
                # b_loss = self.config.balance_loss_coef * b_loss
                # loss += b_loss
                # router_z_loss = [loss_pair[1] for loss_pair in outputs.router_logits]
                # z_loss = sum(router_z_loss) / len(balance_loss)
                # z_loss = self.config.router_z_loss_coef * z_loss
                # loss += z_loss
                # if self.config.local_rank == 0:
                #     print('llm balance loss: ', b_loss.item(), 'llm router z loss: ', z_loss.item())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits=logits,
            loss=loss

        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("pilot", pilotConfig)
AutoModelForCausalLM.register(pilotConfig, pilotLlamaForCausalLM)

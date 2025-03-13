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
# ------------------------------------------------------------------------



import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super().__init__()
        # self.pre_norm = nn.LayerNorm(mm_hidden_size)
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size

        self.proj1 = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.proj2 = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.Diff_loss = DiffLoss()

    def forward(self, x):
        # x = self.pre_norm(x)
        global Diff_loss
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        Diff_loss = Diff_loss(x1, x2)
        x = x1 + x2
        return x, Diff_loss

    @property
    def config(self):
        return {"mm_projector_type": 'two_mlp'}


class MLPMoE(nn.Module):
    def __init__(self, num_experts, num_selected, mm_channels, channels, dropout=False):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.mm_channels = mm_channels
        self.channels = channels

        self.gate = nn.Linear(mm_channels, num_experts, bias=False)
        self.num_selected = num_selected
        self.num_experts = num_experts
        # self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.Diff_loss = DiffLoss()
        self.proj1 = nn.Sequential(
            nn.Linear(mm_channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(mm_channels, channels), nn.GELU(), nn.Linear(channels, channels)) for _ in
             range(num_experts)])

        self.adapters = nn.ModuleList(
            [nn.Sequential(nn.Linear(mm_channels, channels), nn.GELU(), nn.Linear(channels, channels)) for _ in
             range(num_experts - 1)]
        )

    def forward(self, x):
        global Diff_loss
        gate_logits = self.gate(x)
        router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        router_z_loss = torch.square(router_z_loss)
        router_z_loss = router_z_loss.mean()

        x1 = self.proj1(x)
        x2 = self.experts[0](x)
        # log_output1 = F.log_softmax(x1, dim=1)
        # soft_output2 = F.softmax(x2, dim=1)
        # kl_loss = self.kl_loss(log_output1, soft_output2)
        Diff_loss = Diff_loss(x1, x2)
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x.dtype)
        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)

        one_hot_gate_indices = F.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')

        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)

        results = torch.zeros((x.shape[0], x.shape[1], self.channels)).to(x.device, x.dtype)

        for b in range(x.shape[0]):
            for i, expert in enumerate(self.experts):
                token_idx, nth_expert = torch.where(selected_experts[b] == i)
                expert_output = expert(x[b][token_idx])
                if i > 0:
                    adapter_output = self.adapters[i - 1](x[b][token_idx])
                    expert_output += adapter_output
                results[b][token_idx] += weights[b][token_idx, nth_expert, None] * expert_output

        return results, balance_loss, router_z_loss, Diff_loss

    @property
    def config(self):
        return {"mm_projector_type": 'smoe_mlp'}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    elif projector_type == 'two_mlp':
        return SimpleResBlock(mm_hidden_size=config.mm_hidden_size, hidden_size=config.hidden_size)

    elif projector_type == 'smoe_mlp':
        return MLPMoE(num_experts=config.num_experts, num_selected=config.num_selected,
                      # mm_channels=(config.mm_hidden_size * len(config.scales)),
                      mm_channels=config.mm_hidden_size,
                      channels=config.hidden_size, dropout=config.dropout)
    raise ValueError(f'Unknown projector type: {projector_type}')

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class RouterMixin:

    def select_token(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        router_logits = self.router(hidden_states).squeeze(-1) # (batch_size, seq_len)
        if self.top_k == -1: # if -1, defaults to all
            top_k = seq_len
        else:
            top_k = self.top_k

        # routing_weights: (batch_size, seq_len, 1)
        # routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_tokens = torch.topk(router_logits,  top_k, dim=-1, sorted=False)

        # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # routing_weights *= seq_len # need to set to num experts so that if all are used, we get back original result

        # routing_weights = routing_weights.to(hidden_states.dtype) # cast back to hidden states dtype

        routed_output = torch.gather(hidden_states, dim=1, index=selected_tokens.unsqueeze(-1).expand(-1,-1,hidden_dim))
        routed_output = routed_output * routing_weights.unsqueeze(-1).expand_as(routed_output)
        self._router_logits = router_logits # for computing load balancing loss later

        return routed_output


class LinearRouter(nn.Module, RouterMixin):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 1)
        self.top_k = config.mm_router_top_k
        # self.top_p = config.router_top_p
        self._router_logits = None

    def forward(self, image_features: torch.Tensor):
        return self.select_token(image_features)

    @property
    def config(self):
        return {"mm_router_type": 'linear'}

class MLPRouter(nn.Module, RouterMixin):
    def __init__(self, config, mlp_depth: int):
        super().__init__()
        modules = [nn.Linear(config.hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth-1):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        modules.append(nn.GELU())
        modules.append(nn.Linear(config.hidden_size, 1))
        self.router = nn.Sequential(*modules)
        self.top_k = config.mm_router_top_k
        self._router_logits = None

    def forward(self, image_features: torch.Tensor):
        return self.select_token(image_features)

    @property
    def config(self):
        return {"mm_router_type": 'mlp'}


def build_vision_router(config, delay_load=False, **kwargs):
    router_type = getattr(config, 'mm_router_type', 'linear')

    if router_type == 'linear':
        return LinearRouter(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', router_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        return MLPRouter(config, mlp_depth)

    raise ValueError(f'Unknown router type: {router_type}')

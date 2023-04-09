"""
BertLayer中用到，作用为：
升维(768->3072)，激活函数(gelu)；
"""
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # intermediate_size (`int`, *optional*, defaults to 3072)
        # 即 4*hidden_size,4*768=3072
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act] # 这里使用的激活函数是gelu；
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
"""
用于attention计算后；
"""
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        """将hidden_states和input_tensor相加进行LayerNorm，即：残差连接；
        (第一次残差连接)第二次残差连接在：BertOutput；
        """
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
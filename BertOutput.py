"""
BertLayer中用到，作用为：
降维(3072->768)，残差连接；
"""
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 从3072转回768
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        """将hidden_states和input_tensor相加进行LayerNorm，即：残差连接；
        (第二次残差连接)第一次残差连接在：BertSelfOut;
        """
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
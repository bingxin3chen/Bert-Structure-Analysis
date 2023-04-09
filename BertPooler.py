"""
BertModel用到，作用为：
将last_hidden_states切分，转为CLS向量；
[batch, 512, 768]---->[batch, 768]
"""
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """通过简单的获取 the first token 的 hidden_state  来pool模型；
        hidden_states[:,0]中":"对应所有数据，"0"对应所有batch中第一个数据；
        hidden_states.shape = torch.Size([batch, 512, 768])
        hidden_states[:,0].shape = torch.Size([batch, 768])
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output) # 使用的激活函数是Tanh。
        return pooled_output
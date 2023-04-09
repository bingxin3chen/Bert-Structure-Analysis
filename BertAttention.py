class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
#         self.pruned_heads = set() # 剪枝，一般不会用到；
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """进入BertSelfAttention模块，通过QKV计算Attention scores；
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        """进入BertSelfOutput
        self_outputs[0]为"加权后的向量表征"。
        self_outputs[1]为空；
        """
        attention_output = self.output(self_outputs[0], hidden_states) # 其中会进行残差连接；
        # attention_output.shape = torch.Sezie([1,9,768])
        outputs = (attention_output,) + self_outputs[1:]  # self_outputs[1:]是空，不影响；
        return outputs
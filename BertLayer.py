class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

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
        self_attn_past_key_value = None
        """进入BertAttention模块
        只有self_attention_outputs[0]有值，是经过attention计算后加权的向量表征；
        self_attention_outputs[1:]是空。
        """
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # None

        cross_attn_present_key_value = None
        """apply_chunking_to_forward其实为pytorch_utils.py中的一个函数；
        主体为下方的feed_forward_chunk函数；
        """
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        ) 
        # layer_output.shape = torch.Size([1, 9, 768])
        outputs = (layer_output,) + outputs # outputs为空；
        return outputs

    def feed_forward_chunk(self, attention_output):
        """进入BertIntermediate
        """
        intermediate_output = self.intermediate(attention_output) # 升维(768->3072)，激活函数；
        """进入BertOutput模块
        """
        layer_output = self.output(intermediate_output, attention_output) # 降维(3072->768)，残差连接；
        return layer_output
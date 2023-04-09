class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """因为不需要输出所有，所以下方三者全为None。没有删除是因为后面有参数需要填充；
        """
        all_hidden_states = all_self_attentions = all_cross_attentions = None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = past_key_value = None # 没有删除是因为后面有参数需要填充；
            """经过具体的Block结构，也就是循环n次BertLayer模块。(因我使用的是bert-base-chinese，所以n==12)
            layer_module的返回值只有 layer_outputs[0]有值，即：hidden_states，因为我并没有选择
            存储layer_outputs，所以最后一轮返回的hidden_states就是last_hidden_states。
            """
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            """进入BertLayer模块；
            """
            hidden_states = layer_outputs[0] # last_hidden_state
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
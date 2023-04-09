class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        # 不需要输出所有attentions和所有hidden_states，所以设置为False;
        output_attentions = output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        """past_key_values_length
        past_key_values这个参数用来保存前一个片段的Self-Attention输出，将输入切片传入模型训练会用到；
        即：将前一个片段的信息"传递"到当前片段；这里不需要，所以为past_key_values=None，past_key_values_length=0。
        """
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # 我们可以自己提供一个形状为[batch_size, from_seq_length, to_seq_length]的self-attention mask维度，这个可以广播到所有 heads。
        # get_extended_attention_mask 为modeling_utils.py中的函数；(extended  v.拓展；延伸)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_extended_attention_mask = None

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,                                    # None
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask, # None
            past_key_values=past_key_values,                        # None
            use_cache=use_cache,                                    # False
            output_attentions=output_attentions,                    # None
            output_hidden_states=output_hidden_states,              # None
            return_dict=return_dict,
        )
        """
        encoder_outputs[0]为last_hidden_state
        pooled_output为最后一个隐藏层中[CLS]标记的向量表示。
        """
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
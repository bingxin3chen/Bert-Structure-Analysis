import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 设置使用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置多块gpu为 "cuda:0,1,3"

# 加载预训练Bert模型和tokenizer，并将其移动到所选设备上
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

# 如果有多个GPU，将模型包装在nn.DataParallel中
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

# 设置输入并将其移动到所选设备上
input_text = "杨幂同款连衣裙"
inputs = tokenizer(input_text, padding='max_length', max_length=20,return_tensors='pt').to(device)

# 在模型中调试
outputs = model(**inputs)     # 在这里设置断点

# 将输出移动回CPU并查看输出
last_hidden_states = outputs.last_hidden_state.cpu()
pooler_output = outputs.pooler_output.cpu()

# 打印输出形状
print(f"最后一层隐藏层结果 last_hidden_states shape: {last_hidden_states.shape}")
print(f"池化的结果 pooler_output shape: {pooler_output.shape}")
"""
终端打印结果：
最后一层隐藏层结果 last_hidden_states shape: torch.Size([1, 20, 768])
池化的结果 pooler_output shape: torch.Size([1, 768])
"""
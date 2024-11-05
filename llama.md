# Llama 2模型使用指南

## Meta授权申请
1. 访问 [Meta的申请表](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
2. 需要填写详细的个人/公司信息
3. 同意长篇的许可协议（包括使用限制和法律条款）
4. 等待Meta的人工审核，这可能需要几天到几周的时间

⚠️ 重要提醒：
- 申请表中没有中国选项
- Meta会验证IP地址
- 在获得授权之前无法访问模型

## 下载Llama 2
1. 获得Meta授权后，访问 [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)
2. 在Hugging Face上接受使用条款
3. 下载量化版本的模型文件：
   - 对于8GB内存的机器，建议下载 `llama-2-7b-chat.ggmlv3.q4_0.bin`(约3.79GB)
   - 如果内存不足，可以尝试 `llama-2-7b-chat.ggmlv3.q2_K.bin`(约2.87GB)

## 使用示例
```python
model_path = "./llama-2-7b-chat.ggmlv3.q2_K.bin"  # 本地模型文件路径
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

## 注意事项
1. 模型文件较大，下载时需要稳定的网络
2. 运行时建议：
   - 使用量化版本以节省内存
   - 确保有足够的硬盘空间（至少10GB）
   - 关闭其他内存占用大的应用

## 参考资源
- [Llama 2 官方文档](https://ai.meta.com/llama/)
- [Llama 2 使用条款](https://ai.meta.com/llama/license/)
- [TheBloke的量化版本](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) 
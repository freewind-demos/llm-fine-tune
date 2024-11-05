# 大语言模型微调教程

本教程将指导你如何在 MacBook Air M1 8GB 上微调一个小型语言模型。我们将使用 Llama-2-7b-chat 模型的量化版本,因为它相对较小且可以在消费级硬件上运行。

## 环境要求

- MacBook Air M1 (8GB RAM)
- Python 3.9+
- 至少 20GB 可用硬盘空间
- Hugging Face 账号（用于下载模型）

## 步骤一：环境准备

1. 打开终端，安装 Homebrew (如果还没安装):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. 安装 Python (如果还没安装):
```bash
brew install python@3.9
```

3. 创建并进入项目目录:
```bash
mkdir llm-finetune
cd llm-finetune
```

4. 创建虚拟环境:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 步骤二：安装必要的包

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate bitsandbytes
pip install -U loralib
pip install sentencepiece
```

## 步骤三：下载模型

1. 访问 [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) 模型页面

2. 点击页面顶部的 "Files" 标签页

3. 根据你的硬件配置选择合适的模型文件:
   - 对于8GB内存的M1 MacBook Air,建议下载 `llama-2-7b-chat.ggmlv3.q4_0.bin`(约3.79GB)
   - 如果遇到内存不足,可以尝试 `llama-2-7b-chat.ggmlv3.q2_K.bin`(约2.87GB)

4. 点击文件名右侧的下载图标即可开始下载

5. 将下载的模型文件放到项目目录下

6. 确保在运行微调脚本之前，将模型文件名更新到 `finetune.py` 中的 `model_path` 变量：
```python
model_path = "./llama-2-7b-chat.ggmlv3.q2_K.bin"  # 确保这里的文件名与你下载的模型文件名一致
```

注意:
- 首次访问需要登录 Hugging Face 账号
- 需要先在 [Meta的申请表](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) 申请使用 Llama 2
- 在 Hugging Face 上接受使用条款

## 步骤四：准备训练数据

1. 创建训练数据文件 `train_data.json`：

```bash
touch train_data.json
```

2. 将以下示例数据复制到 `train_data.json` (可以根据需要修改或添加更多数据):

```json
[
    {
        "instruction": "介绍一下北京",
        "input": "",
        "output": "北京是中国的首都，是一座具有悠久历史文化的城市。这里有故宫、长城等著名景点。"
    },
    {
        "instruction": "写一首关于春天的诗",
        "input": "",
        "output": "春风轻抚柳枝摆，花朵绽放满园开。蝴蝶翩翩舞青空，春色十里醉人怀。"
    }
]
```

## 步骤五：运行微调脚本

1. 创建训练脚本 `finetune.py`:

```bash
touch finetune.py
```

2. 将以下代码复制到 `finetune.py`:

```python
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch
from peft import get_peft_model, LoraConfig, TaskType

# 加载模型和分词器
model_path = "./llama-2-7b-chat.ggmlv3.q2_K.bin"  # 本地模型文件路径
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# 应用 LoRA
model = get_peft_model(model, peft_config)

# 加载数据集
dataset = load_dataset("json", data_files="train_data.json")

# 数据预处理函数
def preprocess_function(examples):
    text = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:
            text.append(f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}")
        else:
            text.append(f"Instruction: {instruction}\nOutput: {output}")
    
    return tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

# 预处理数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=50,
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("./fine_tuned_model")
```

## 步骤六：开始训练

在终端中运行:

```bash
python finetune.py
```

训练过程大约需要 1-2 小时，具体时间取决于你的训练数据量。

## 步骤七：使用微调后的模型

训练完成后，你可以使用以下代码测试模型:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载微调后的模型
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")

# 测试模型
def generate_response(instruction):
    inputs = tokenizer(f"Instruction: {instruction}", return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例使用
response = generate_response("介绍一下上海")
print(response)
```

## 注意事项

1. 确保你的 MacBook 在训练过程中接通电源。
2. 训练过程中电脑可能会发热，这是正常现象。
3. 如果遇到内存不足的问题，可以：
   - 减少 batch_size
   - 减少模型最大序列长度
   - 关闭其他占用内存的应用

## 常见问题解决

1. 如果安装包时出现错误，尝试：
```bash
pip install --upgrade pip
```

2. 如果运行时出现 CUDA 相关错误，不用担心，因为 M1 芯片使用 MPS 而不是 CUDA。

3. 如果出现内存错误，可以修改 `finetune.py` 中的这些参数：
```python
per_device_train_batch_size=2  # 减小批次大小
max_length=256  # 减小最大序列长度
```

## 进阶建议

1. 当你熟悉了基本的微调流程后，可以尝试：
   - 使用更多的训练数据
   - 调整学习率和训练轮数
   - 尝试不同的模型架构

2. 为了获得更好的效果，建议：
   - 准备高质量的训练数据
   - 仔细设计指令格式
   - 实验不同的超参数组合

## 参考资源

- [Hugging Face 文档](https://huggingface.co/docs)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Llama 2 官方文档](https://ai.meta.com/llama/)
# 大语言模型微调教程

本教程将指导你如何在 MacBook Air M1 8GB 上微调一个小型语言模型。我们将使用 gpt2-chinese-cluecorpussmall 模型，因为它体积小（约400MB）且可以在消费级硬件上运行。

## 环境要求

- MacBook Air M1 (8GB RAM)
- Python3.9 或 3.10（不要使用 3.13，因为很多包还不支持）
- 至少 20GB 可用硬盘空间

## 步骤一：环境准备

1. 打开终端，安装 Homebrew (如果还没安装):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. 安装 Python3.9:
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
python3.9 -m venv venv
source venv/bin/activate
```

## 步骤二：安装必要的包

1. 首先安装基础工具：
```bash
brew install cmake pkg-config
pip install --upgrade pip setuptools wheel
```

2. 安装 PyTorch（对于 M1 Mac 的特殊命令）：
```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

3. 安装 Hugging Face 相关包：
```bash
pip install transformers
pip install datasets
pip install accelerate
```

4. 安装其他必要的包：
```bash
pip install bitsandbytes
pip install -U loralib
pip install sentencepiece
pip install peft
```

5. 验证安装：

分步验证每个包的安装：
```bash
# 验证 torch
python3.9 -c "import torch; print('PyTorch version:', torch.__version__)"

# 验证 transformers
python3.9 -c "import transformers; print('Transformers version:', transformers.__version__)"

# 验证 datasets
python3.9 -c "import datasets; print('Datasets version:', datasets.__version__)"

# 验证 peft
python3.9 -c "import peft; print('PEFT version:', peft.__version__)"
```

每条命令都应该打印出相应包的版本号。如果某个命令失败，说明该包安装有问题。

注意：首次运行时，transformers 可能会下载一些配置文件，这可能需要一些时间。如果卡住太久，可以按 Ctrl+C 中断，然后重试。

如果看到"所有包都已正确安装！"的消息，就说明环境准备好了。如果出现错误，请检查错误信息并重新安装相应的包。

## 步骤三：下载模型

1. 安装 git-lfs：
```bash
brew install git-lfs
git lfs install
```

2. 下载模型（如果之前的方法失败，使用这个替代方案）：
```bash
# 创建模型目录
mkdir -p gpt2-chinese-cluecorpussmall
cd gpt2-chinese-cluecorpussmall

# 直接下载必要的文件
wget https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/resolve/main/config.json
wget https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/resolve/main/pytorch_model.bin
wget https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/resolve/main/tokenizer_config.json
wget https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/resolve/main/vocab.txt
wget https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/resolve/main/special_tokens_map.json

# 如果wget没有安装，先安装它
# brew install wget
```

3. 验证下载：
```bash
# 检查文件是否都下载完整
ls -lh
```

应该看到以下文件：
- config.json（模型配置文件）
- pytorch_model.bin（模型权重文件，约400MB）
- tokenizer_config.json（分词器配置）
- vocab.txt（词汇表）
- special_tokens_map.json（特殊token映射）

4. 模型信息：
   * 约400MB，最小巧的中文预训练模型之一
   * 专门针对中文优化
   * 训练数据来自清华CLUE项目
   * 完全开源，无需注册即可使用

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

3. 如果要将代码文件添加到训练数据中：

首先创建代码转换脚本 `convert_code_to_data.py`:
```bash
touch convert_code_to_data.py
```

将上面提供的代码复制到 `convert_code_to_data.py` 中，然后运行：
```bash
python3.9 convert_code_to_data.py
```

这将把 `data-code/hello.ts` 文件转换并添加到 `train_data.json` 中。如果要添加其他代码文件，可以修改脚本中的 `code_files` 列表。

转换后的训练数据格式如下：
```json
[
    // 原有的训练数据...
    {
        "instruction": "请实现一个hello.ts文件",
        "input": "",
        "output": "export function helloFreewind() {\n    console.log(\"Hello Freewind\");\n}\n"
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

# 使用中文GPT2小模型
model_path = "./gpt2-chinese-cluecorpussmall"  # 本地模型目录
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
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
python3.9 finetune.py
```

训练过程大约需要 1-2 小时，具体时间取决于你的训练数据量。

## 步骤七：使用微调后的模型

训练完成后，创建`try-tuned-model.py`文件来测试模型：

```bash
touch try-tuned-model.py
```

将以下代码复制到`try-tuned-model.py`中：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载微调后的模型和tokenizer
model_path = "./fine_tuned_model"
original_model_path = "./gpt2-chinese-cluecorpussmall"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(original_model_path)  # 使用原始模型的tokenizer

# 测试模型
def generate_response(instruction):
    # 构造输入
    prompt = f"Instruction: {instruction}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成回答
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除输入提示，只返回生成的部分
    response = response.replace(prompt, "").strip()
    return response

# 示例使用
print("测试1：介绍北京")
response = generate_response("介绍一下北京")
print(response)

print("\n测试2：生成代码")
response = generate_response("请实现一个 hello.ts 文件")
print(response)
```

运行测试脚本：
```bash
python3.9 try-tuned-model.py
```

## 注意事项

1. 确保你的 MacBook 在训练过程中接通电源。
2. 训练过程中电脑可能会发热，这是正常现象。
3. 如果遇到内存不足的问题，可以：
   - 减少 batch_size
   - 减少模型最大序列长度
   - 关闭其他占用内存的应用
4. 在 M1 Mac 上：
   - 不支持 GPU 相关功能（如混合精度训练）
   - 使用 MPS 后端代替 CUDA
   - 某些高级优化功能可能不可用

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
- [GPT2-Chinese-Small模型](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)
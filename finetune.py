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

# 加载预训练模型和分词器
model_path = "./gpt2-chinese-cluecorpussmall"  # 指定预训练模型的本地路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用float16进行混合精度训练，可以减少显存占用
    device_map="auto",  # 自动决定模型加载到CPU还是GPU
)

# 配置 LoRA (Low-Rank Adaptation) 参数
# LoRA 是一种参数高效的微调方法，只训练一小部分参数，大大减少显存占用
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
    inference_mode=False,  # 训练模式
    r=8,  # LoRA 的秩，越大效果越好，但参数量也越大
    lora_alpha=32,  # LoRA的缩放参数，通常设置为 r 的 4 倍
    lora_dropout=0.1,  # LoRA的dropout率，用于防止过拟合
)

# 将 LoRA 应用到模型
model = get_peft_model(model, peft_config)

# 加载自定义的训练数据集
dataset = load_dataset("json", data_files="train_data.json")

# 将数据集分割为训练集和评估集
# test_size=0.2 表示评估集占总数据的 20%
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 定义数据预处理函数
def preprocess_function(examples):
    text = []
    # 将instruction（指令）、input（输入）和output（输出）组合成特定格式
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # 如果有输入文本，使用完整格式；否则使用简化格式
        if input_text:
            text.append(f"问：{instruction}\n内容：{input_text}\n答：{output}\n")
        else:
            text.append(f"问：{instruction}\n答：{output}\n")
    
    # 使用tokenizer进行编码
    encoded = tokenizer(
        text,
        truncation=True,  # 截断超长文本
        max_length=512,  # 最大序列长度
        padding="max_length",  # 填充到最大长度
        return_special_tokens_mask=True,  # 返回特殊token的mask
    )
    return encoded

# 对训练集和评估集进行预处理
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,  # 批处理模式
    remove_columns=train_dataset.column_names,  # 移除原始列
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 输出目录
    num_train_epochs=50,  # 训练轮数
    per_device_train_batch_size=1,  # 每个设备的训练批次大小
    save_steps=20,  # 每20步保存一次模型
    logging_steps=5,  # 每5步记录一次日志
    learning_rate=5e-5,  # 学习率
    warmup_steps=100,  # 预热步数，在这些步数内学习率会从0逐渐增加到设定值
    gradient_accumulation_steps=8,  # 梯度累积步数，用于模拟更大的批次大小
)

# 创建训练器
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=tokenized_train_dataset,  # 训练数据集
    eval_dataset=tokenized_eval_dataset,  # 评估数据集
    # 数据整理器，用于将数据组织成模型所需的格式
    # mlm=False 表示不使用掩码语言模型（即使用普通的因果语言模型）
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("./fine_tuned_model")
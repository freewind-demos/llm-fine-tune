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

# 分割数据集为训练集和评估集
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 数据预处理函数
def preprocess_function(examples):
    text = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # 使用更简单的中文格式，避免特殊字符
        if input_text:
            text.append(f"问：{instruction}\n内容：{input_text}\n答：{output}\n")
        else:
            text.append(f"问：{instruction}\n答：{output}\n")
    
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_special_tokens_mask=True,
    )
    return encoded

# 预处理数据集
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# 训练参数调整
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,  # 大幅增加训练轮数
    per_device_train_batch_size=1,  # 减小batch size
    save_steps=20,
    logging_steps=5,
    learning_rate=5e-5,  # 进一步降低学习率
    warmup_steps=100,
    gradient_accumulation_steps=8,  # 增加梯度累积
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # 添加评估数据集
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("./fine_tuned_model") 
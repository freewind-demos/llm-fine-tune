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
model_name = "TheBloke/Llama-2-7B-Chat-GGML"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
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
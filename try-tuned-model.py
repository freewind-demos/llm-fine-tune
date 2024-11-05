from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载微调后的模型和tokenizer
model_path = "./fine_tuned_model"
original_model_path = "./gpt2-chinese-cluecorpussmall"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(original_model_path)

def generate_response(instruction):
    # 构造输入
    prompt = f"问：{instruction}\n答："
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成回答
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,  # 减小最大长度
        num_return_sequences=1,
        do_sample=True,
        temperature=0.3,  # 进一步降低温度使输出更稳定
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.5,  # 增加重复惩罚
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        length_penalty=1.5,  # 添加长度惩罚
        early_stopping=True
    )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response.replace(" ", "")  # 移除空格

# 示例使用
test_cases = [
    "介绍一下北京",
    "写一首关于春天的诗",
    "请实现一个hello.ts文件"
]

for test in test_cases:
    print(f"\n测试：{test}")
    response = generate_response(test)
    print(f"回答：{response}")
    print("-" * 50)
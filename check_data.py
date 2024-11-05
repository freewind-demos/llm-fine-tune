from datasets import load_dataset

# 加载数据集
dataset = load_dataset("json", data_files="train_data.json")

# 打印所有训练数据
print("训练数据集中的所有数据：")
for item in dataset["train"]:
    print("\n---")
    print(f"Instruction: {item['instruction']}")
    print(f"Input: {item['input']}")
    print(f"Output: {item['output']}") 
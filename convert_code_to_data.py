import json
import os

def convert_code_to_training_data(code_file_path):
    """将代码文件转换为训练数据格式"""
    with open(code_file_path, 'r') as f:
        code_content = f.read()
    
    # 获取文件名和扩展名
    filename = os.path.basename(code_file_path)
    
    return {
        "instruction": f"请实现一个 {filename} 文件",
        "input": "",
        "output": code_content
    }

def update_training_data(code_files, train_data_path="train_data.json"):
    """更新训练数据文件"""
    # 读取现有的训练数据
    try:
        with open(train_data_path, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # 转换每个代码文件并添加到训练数据中
    for code_file in code_files:
        training_item = convert_code_to_training_data(code_file)
        existing_data.append(training_item)
    
    # 保存更新后的训练数据
    with open(train_data_path, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 指定要添加的代码文件
    code_files = ["data-code/hello.ts"]
    update_training_data(code_files) 
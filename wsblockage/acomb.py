import os

def sum_columns(file_paths, output_file):
    """
    读取多个文件，每个文件是一列数据，将它们逐列累加后保存为一个新的文件。
    
    Args:
        file_paths (list): 输入文件的路径列表
        output_file (str): 输出文件的路径
    """
    # 检查所有文件是否存在
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，程序终止。")
            return

    # 读取第一个文件以获取行数
    with open(file_paths[0], 'r') as f:
        lines = f.readlines()
    num_rows = len(lines)

    # 初始化结果列表
    result = [0.0] * num_rows

    # 逐行累加每个文件的数据
    for file_path in file_paths:
        print(f"正在处理文件 {file_path}...")
        with open(file_path, 'r') as f:
            current_data = f.readlines()
            # 检查文件行数是否一致
            if len(current_data) != num_rows:
                print(f"文件 {file_path} 的行数与第一个文件不一致，程序终止。")
                return
            # 将当前文件的每一行转换为浮点数并累加到结果中
            for i in range(num_rows):
                try:
                    value = float(current_data[i].strip())
                    result[i] += value
                except ValueError:
                    print(f"文件 {file_path} 的第 {i+1} 行不是有效的数字，程序终止。")
                    return

    # 将结果保存到输出文件
    with open(output_file, 'w') as f:
        for value in result:
            f.write(f"{value}\n")
    print(f"累加结果已保存到 {output_file}")

# 主程序
if __name__ == "__main__":
    file_paths = [f"C{i}day1M.txt" for i in range(1, 451)]
    output_file = "sum_450.txt"

    sum_columns(file_paths, output_file)
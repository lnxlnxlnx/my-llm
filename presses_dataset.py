def process_text_file(input_file_path, output_file_path):
    """
    处理txt文件中的数据，转换为指定格式
    :param input_file_path: 原始数据文件路径
    :param output_file_path: 处理后数据保存路径
    """
    processed_lines = []

    # 读取原始文件
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_file_path}")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return

    # 逐行处理数据
    for line_num, line in enumerate(lines, 1):
        # 去除行首尾的空白字符（换行、空格、制表符等）
        clean_line = line.strip()
        if not clean_line:  # 跳过空行
            continue

        # 分割文件名和内容（按 ", " 分割，兼容带空格的情况）
        if ', "' in clean_line:
            filename_part, content_part = clean_line.split(', "', 1)
            # 去除内容部分末尾的双引号
            content = content_part.rstrip('"').strip()
            # 拼接为 文件名\t内容 格式
            processed_line = f"{filename_part.strip()}\t{content}"
            processed_lines.append(processed_line)
        else:
            # 处理格式异常的行（可选：提示并跳过）
            print(f"警告：第 {line_num} 行格式异常，已跳过 → {clean_line}")

    # 保存处理后的数据到文件
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(processed_lines))
        print(f"\n处理完成！共处理 {len(processed_lines)} 行有效数据")
        print(f"处理后的数据已保存到：{output_file_path}")
    except Exception as e:
        print(f"保存文件时出错：{e}")
        return

    # 可选：打印前10行结果预览
    print("\n处理结果预览（前10行）：")
    for _, line in enumerate(processed_lines[:10], 1):
        print(line)


# ===================== 配置文件路径 =====================
# 请修改这里的文件路径为你的实际路径
INPUT_FILE = "new_labels.txt"  # 你的原始数据txt文件路径
OUTPUT_FILE = "labels.txt"  # 处理后保存的文件路径
# ========================================================

# 执行处理
if __name__ == "__main__":
    process_text_file(INPUT_FILE, OUTPUT_FILE)

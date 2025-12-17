# 生成256点、10位精度（0~1023）的正弦波MIF文件
depth = 256  # 存储深度（1个周期的点数）
width = 10   # 数据宽度（DAC为10位）
output_file = "sine_wave.mif"

with open(output_file, "w", encoding="utf-8") as f:
    # 写入MIF文件头
    f.write(f"DEPTH = {depth};\n")
    f.write(f"WIDTH = {width};\n")
    f.write("ADDRESS_RADIX = DEC;\n")  # 地址用十进制
    f.write("DATA_RADIX = DEC;\n")     # 数据用十进制
    f.write("CONTENT BEGIN\n")

    # 计算并写入每个地址的正弦数据
    import math
    for i in range(depth):
        theta = 2 * math.pi * i / depth  # 相位（0~2π）
        sin_val = 4 * math.sin(theta)        # 正弦值（-1~1）
        mapped_val = round((sin_val + 1) * 511)  # 映射为0~1023
        f.write(f"    {i} : {mapped_val};\n")  # 地址:数据

    f.write("END;\n")
print(f"MIF文件已生成：{output_file}")

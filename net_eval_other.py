import argparse
import warnings
import torch
import socket
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *

# 忽略无关警告
warnings.filterwarnings("ignore")


def clean_k230_response(raw_responses):
    """
    清洗K230响应数据：保留中文+关键英文+数字，过滤乱码符号
    """
    # 合并分段响应
    merged_text = " ".join(raw_responses)
    print(f"\n📥 K230原始合并数据: {merged_text}")

    # 核心清洗规则：保留中文、英文、数字、常见标点，过滤乱码/特殊符号
    # 保留范围：中文(\u4e00-\u9fa5)、英文(a-zA-Z)、数字(0-9)、关键符号(|:.,()_)、常见标点
    cleaned_text = re.sub(
        r'[^\u4e00-\u9fa5a-zA-Z0-9|:.,()_\s，。！？]',
        '',
        merged_text
    )
    # 替换竖线为空格，合并多余空格
    cleaned_text = cleaned_text.replace("|", " ").strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # 提取关键关键词（兜底）
    keyword_pattern = r"(except|socket|print|connect|recv|send|client|你是谁|\d+)"
    keywords = re.findall(keyword_pattern, cleaned_text.lower())
    keywords = list(set(keywords)) if keywords else []

    # 确保清洗后有有效内容
    if not cleaned_text and keywords:
        cleaned_text = " ".join(keywords)
    if not cleaned_text:
        cleaned_text = "未识别到有效信息，仅检测到乱码"

    print(f"🧹 清洗后有效文本: {cleaned_text}")
    return cleaned_text


def collect_and_clean_k230_data(server_ip, server_port, timeout=5):
    """
    完整交互：连接K230+发送数据+接收响应+清洗数据
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(timeout)
    raw_responses = []

    try:
        # 1. 连接K230服务端
        print(f"🔌 尝试连接K230 [{server_ip}:{server_port}]...")
        client_socket.connect((server_ip, server_port))
        print("✅ 成功连接K230服务端！")

        # 2. 发送数据（匹配你测试代码的发送逻辑）
        send_data = f"PC客户端消息: 当前时间 {time.time():.0f}".encode()
        client_socket.send(send_data + b"\n")
        print(f"📤 已发送数据: {send_data.decode()}")

        # 3. 接收K230响应（分段接收）
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                response = chunk.decode("utf-8", errors="ignore").strip()
                if response:
                    if response == "clash":
                        print("😭服务端不准备发送数据！")
                        time.sleep(3)
                        return "", ""
                    raw_responses.append(response)
                    print(f"🌟 收到K230响应: {response}")
            except socket.timeout:
                break

        # 4. 清洗数据
        cleaned_text = clean_k230_response(raw_responses)
        return cleaned_text

    except socket.timeout:
        print("❌ 连接/接收超时！请检查K230服务端状态")
        return ""
    except ConnectionRefusedError:
        print("❌ 连接被拒绝！请确认K230服务端已启动")
        return ""
    except Exception as e:
        print(f"❌ K230交互异常: {e}")
        return ""
    finally:
        client_socket.close()
        print("🔌 已关闭K230连接")


def init_model(args):
    """
    初始化LLM模型：修复参数名称+适配中文
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # 适配中文：补充缺失的token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "model" in args.load_from:
        model = MiniMindForCausalLM(
            MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        # 加载权重并修复参数名称
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        new_state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(new_state_dict, strict=True)

        # 加载LoRA（如果指定）
        if args.lora_weight != "None":
            apply_lora(model)
            load_lora(model, f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    # 打印模型参数信息
    param_num = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n🔧 MiniMind模型加载完成 | 参数总量: {param_num:.2f} M")
    return model.eval().to(args.device), tokenizer


def main():
    # 参数解析（精简无用参数）
    parser = argparse.ArgumentParser(description="MiniMind + K230 交互推理")
    # LLM核心参数
    parser.add_argument("--load_from", default="model", type=str, help="模型加载路径")
    parser.add_argument("--save_dir", default="out", type=str, help="权重目录")
    parser.add_argument("--weight", default="pretrain", type=str, help="权重前缀")
    parser.add_argument("--lora_weight", default="None", type=str, help="LoRA权重")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0,1], help="是否MoE架构")
    parser.add_argument("--inference_rope_scaling", default=False, action="store_true", help="RoPE外推")
    # 生成参数
    parser.add_argument("--max_new_tokens", default=512, type=int, help="最大生成长度")
    parser.add_argument("--temperature", default=0.7, type=float, help="生成温度")
    parser.add_argument("--top_p", default=0.8, type=float, help="Top-P采样")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    # K230连接参数
    parser.add_argument("--k230_ip", default="192.168.41.134", type=str, help="K230 IP")
    parser.add_argument("--k230_port", default=8888, type=int, help="K230 端口")
    args = parser.parse_args()

    # 1. 初始化LLM模型
    model, tokenizer = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    while True:
        # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client_socket.settimeout(3)
        # print(f"🔌 尝试连接K230 [{args.server_ip}:{args.server_port}]...")
        # client_socket.connect((args.server_ip, args.server_port))
        # print("✅ 成功连接K230服务端！")
        # 2. 与K230交互，获取并清洗数据
        k230_text = collect_and_clean_k230_data(args.k230_ip, args.k230_port)
        if not k230_text:
            print("\n❌ 无有效K230数据，退出推理")
            return

        # 3. 构建LLM输入Prompt（适配交流场景）
        prompt = f"""
        请分析以下从K230设备获取的信息，并回答相关问题：
        设备返回内容：{k230_text}
        
        要求：
        1. 识别其中的关键信息（包括中文问题、错误关键词）；
        2. 用自然语言回答其中的问题，解释错误信息（如果有）；
        3. 回复语言为中文，简洁易懂。
        """.strip()

        # 4. 编码输入
        conversation = [{"role": "user", "content": prompt}]
        inputs_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ) if args.weight != "pretrain" else tokenizer.bos_token + prompt
        
        inputs = tokenizer(
            inputs_text, return_tensors="pt", truncation=True, padding=True
        ).to(args.device)

        # 5. LLM生成回复
        print("\n🤖️ LLM正在生成回复...")
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.05,  # 中文去重
        )

        # 6. 解码并输出回复
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # print(f"\n✅ 最终回复:\n{response}")


if __name__ == "__main__":
    main()
    
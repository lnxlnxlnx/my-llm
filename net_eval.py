import argparse
import warnings
import torch
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import time
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
import re

# 忽略无关警告
warnings.filterwarnings("ignore")


def clean_k230_response(raw_responses):
    """
    清洗K230响应数据，提取有效信息
    :param raw_responses: 原始分段响应列表（如["Ora | l | Qn | ece口 |", "Orai |
    | Qm | except | print | except | 124221"]）
    :return: 清洗后的有效文本、提取的关键词列表
    """
    # 步骤1：合并所有分段响应
    merged_text = " ".join(raw_responses)
    print(f"📥 原始合并数据: {merged_text}")

    # 步骤2：清洗规则（按优先级过滤）
    # 2.1 替换特殊分隔符/乱码字符
    replace_rules = {
        r"\|": " ",  # 替换竖线为空格
        # r"▲|筐|旗|仙|叁|凌|叉发|址姓|a姓|Ｇ|Ln|Co": "",  # 过滤乱码/特殊字符
        r"[^\x00-\x7F\u4e00-\u9fa5]": "",  # 过滤非ASCII+非中文的乱码
        r"\s+": " ",  # 多个空格合并为一个
    }
    cleaned_text = merged_text
    for pattern, repl in replace_rules.items():
        cleaned_text = re.sub(pattern, repl, cleaned_text)

    # 步骤3：提取有效关键词（如except、print、数字、变量名等）
    # 匹配规则：字母+数字组合、except/print/orintt/printt等关键词、纯数字
    keyword_pattern = r"(except|print|orintt|printt|\w+\d+|\d+)"
    keywords = re.findall(keyword_pattern, cleaned_text.lower())  # 转小写统一格式
    keywords = list(set(keywords))  # 去重

    # 步骤4：最终文本标准化（去除首尾空格，补充上下文）
    cleaned_text = cleaned_text.strip()
    # 如果清洗后文本为空，用关键词兜底
    if not cleaned_text and keywords:
        cleaned_text = " ".join(keywords)

    print(f"🧹 清洗后文本: {cleaned_text}")
    print(f"🔑 提取的关键词: {keywords}")
    return cleaned_text, keywords


def collect_k230_responses(server_ip, server_port, timeout=3):
    """
    连接服务端，收集所有K230分段响应
    :param server_ip: 服务端IP
    :param server_port: 服务端端口
    :param timeout: 超时时间（秒）
    :return: 原始响应列表、清洗后的有效文本
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(timeout)  # 设置超时，避免无限等待
    raw_responses = []

    try:
        client_socket.connect((server_ip, server_port))
        print(f"✅ 成功连接服务端 {server_ip}:{server_port}")

        start_time = time.time()
        # 循环接收所有分段响应，直到超时或无数据
        while time.time() - start_time < timeout:
            try:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                # 解码并去除结束标记（如果有）
                response = (
                    chunk.decode("utf-8", errors="ignore").replace("|END", "").strip()
                )
                if response:  # 非空响应才收集
                    raw_responses.append(response)
                    print(f"🌟 收到K230响应: {response}")
            except socket.timeout:
                break

        # 清洗数据
        cleaned_text, keywords = clean_k230_response(raw_responses)
        # return raw_responses, cleaned_text
        return cleaned_text, keywords

    except Exception as e:
        print(f"❌ 连接/接收数据失败: {e}")
        return [], ""
    finally:
        client_socket.close()


def extract_valid_question(raw_text: str) -> str:
    """
    从原始OCR日志里提取真正要提问的那一行。
    返回空串表示没有提取到有效内容。
    """
    if not raw_text:
        return ""

    # 只保留“识别结果”那一行
    lines = raw_text.splitlines()
    for line in lines:
        # 匹配“识别结果 #数字: 内容”
        m = re.match(r"🌟 收到K230响应:\s*", line)
        if m:
            content = m.group(0)[len("🌟 收到K230响应:") - 1 :]
            # 再过滤一次，去掉乱七八糟的符号，只保留中英文、数字、常见标点
            content = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。！？、()\s]", "", content)
            return content
    return ""


# ====================== 原有LLM模型初始化逻辑（完全保留） ======================
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if "model" in args.load_from:
        model = MiniMindForCausalLM(
            MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"

        # 修复参数名称不匹配问题（原有逻辑保留）
        state_dict = torch.load(ckp, map_location=args.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        new_state_dict = {}
        for key, value in state_dict.items():
            if "self_attention" in key:
                new_key = key.replace("self_attention", "self_attn")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=True)

        if args.lora_weight != "None":
            apply_lora(model)
            load_lora(
                model,
                f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )
    print(
        f"MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)"
    )
    return model.eval().to(args.device), tokenizer


# ====================== Socket客户端（接收服务端数据） ======================
# -not used
def connect_server(server_ip, server_port):
    """
    作为客户端连接服务端，接收服务端发送的数据
    :return: 服务端返回的文本数据（去除结束标记）
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 连接服务端
        client_socket.connect((server_ip, server_port))
        print(f"✅ 成功连接服务端 {server_ip}:{server_port}")

        # 接收服务端数据（按结束标记分割）
        recv_data = b""
        while True:
            chunk = client_socket.recv(1024)
            if not chunk:
                break
            recv_data += chunk
            if b"|END" in recv_data:  # 匹配服务端的结束标记
                break

        # 解码并去除结束标记
        server_text = recv_data.decode("utf-8").replace("|END", "").strip()
        print(f"📥 从服务端接收数据: {server_text}")
        return server_text

    except Exception as e:
        print(f"❌ 连接服务端失败: {e}")
        return None
    finally:
        client_socket.close()
        print("🔌 客户端连接已关闭")


# ====================== 主函数（核心逻辑：客户端收数据 → LLM生成回复） ======================
def main():
    # 原有LLM参数解析（新增服务端IP/端口参数）
    parser = argparse.ArgumentParser(
        description="MiniMind模型客户端：接收服务端数据并生成回复"
    )
    # LLM模型参数（完全保留你原有配置）
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument(
        "--weight",
        default="m_pretrain",
        type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）",
    )
    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=8,
        type=int,
        help="隐藏层数量（Small/MoE=8, Base=16）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=False,
        action="store_true",
        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=8192,
        type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )
    parser.add_argument(
        "--historys",
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    # Socket客户端参数（新增）
    parser.add_argument(
        "--server_ip",
        default="127.0.0.1",  # 默认连接本地服务端
        type=str,
        help="服务端IP地址",
    )
    parser.add_argument("--server_port", default=8888, type=int, help="服务端端口号")
    args = parser.parse_args()

    # 1. 初始化LLM模型和分词器
    print("===== 初始化MiniMind模型 =====")
    model, tokenizer = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 2. 作为客户端连接服务端，接收数据
    server_ip_ = "192.168.41.134"
    server_port_ = 8888
    print("\n===== 连接服务端接收数据 =====")
    # server_text = connect_server(server_ip=server_ip_, server_port=server_port_)
    while True:
        _, keywords = collect_k230_responses(server_ip_, server_port_, 3)
        # server_text=server_text.strip()
        server_text = "".join(keywords)
        if not server_text:
            print("❌ 未从服务端获取到有效数据，程序退出")
            continue
        break
        # return

    # 3. 把清洗后的内容送给LLM
    print(f"\n===== 提取到有效提问：{server_text} =====")
    conversation = [{"role": "user", "content": server_text}]
    # 原有prompt构建逻辑完全保留
    templates = {
        "conversation": conversation,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.weight == "reason":
        templates["enable_thinking"] = True  # 仅Reason模型使用
    inputs = (
        tokenizer.apply_chat_template(**templates)
        if args.weight != "pretrain"
        else (tokenizer.bos_token + server_text)
    )
    inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

    # 生成并输出回复
    print("🤖️ LLM回复: ", end="")
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=1.0,
    )
    # 解码完整回复（可选）
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    print(f"\n\n✅ 回复完成: {response}")


if __name__ == "__main__":
    main()

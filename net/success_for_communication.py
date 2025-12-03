import socket
import time


def connect_k230_server():
    # K230服务端的IP和端口（替换为实际K230的IP）
    K230_IP = "192.168.41.134"  # K230的实际IP
    K230_PORT = 8888
    client_socket = None

    try:
        # 1. 创建TCP客户端Socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)  # 连接超时时间

        # 2. 连接K230服务端
        print(f"尝试连接K230服务端 [{K230_IP}:{K230_PORT}]...")
        client_socket.connect((K230_IP, K230_PORT))
        print("✅ 成功连接K230服务端！")

        # 3. 向K230发送数据🌟
        send_data = f"PC客户端消息: 当前时间 {time.time():.0f}".encode()
        client_socket.send(send_data + b"\n")
        print(f"🌟 已发送数据: {send_data.decode()}")

        # 4. 接收K230的响应
        recv_data = client_socket.recv(1024)
        if recv_data:
            print(f"🌟 收到K230响应: {recv_data.decode().strip()}")

    except socket.timeout:
        print("❌ 连接超时！请检查K230服务端是否启动或IP/端口是否正确")
    except ConnectionRefusedError:
        print("❌ 连接被拒绝！请确认K230服务端已启动且端口正确")
    except Exception as e:
        print(f"❌ 连接异常: {e}")
    finally:
        # 5. 关闭连接
        if client_socket:
            client_socket.close()
            print("🔌 连接已关闭")


if __name__ == "__main__":
    connect_k230_server()

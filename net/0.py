# 服务端代码（修复版）
import socket
import time


def start_server(ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 允许端口复用（避免重启服务器时出现 "address already in use" 错误）
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(5)  # 增大监听队列，支持多个客户端排队
    print(f"服务器正在 {ip}:{port} 上监听...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"连接地址：{addr}")

        try:
            # 发送数据（可加上结束标记，方便客户端判断接收完成）
            client_socket.send(b"Hello from Server!|END")  # 加结束标记
            print("数据发送完成，等待客户端接收...")
            time.sleep(2)  # 延迟2秒再关闭，给客户端足够时间接收
        except Exception as e:
            print(f"发送数据出错：{e}")
        finally:
            client_socket.close()  # 确保最终关闭连接
            print(f"客户端 {addr} 连接已关闭\n")


# 启动服务器（确保该IP是服务器实际局域网IP）
start_server("0.0.0.0", 8888)

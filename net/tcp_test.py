import socket


def pc_tcp_server():
    # 配置服务器地址（必须和设备端一致！）
    HOST = "0.0.0.0"  # 监听所有网卡（允许局域网内所有设备连接）
    PORT = 8888  # 和设备端的 PC_PORT 保持一致（8888）

    # 创建 TCP 服务器 Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许端口复用
    server_socket.bind((HOST, PORT))  # 绑定端口
    server_socket.listen(5)  # 最大监听 5 个连接

    print(f"💻 电脑 TCP 服务器启动成功！监听 {HOST}:{PORT}")
    print("等待设备连接...（按 Ctrl+C 退出）\n")

    try:
        while True:
            # 等待设备连接（阻塞直到有设备接入）
            client_conn, client_addr = server_socket.accept()
            print(f"📞 新设备接入: {client_addr}（设备 IP 和端口）")

            try:
                client_conn.settimeout(10)  # 10 秒无数据则断开
                while True:
                    # 接收设备发送的数据（最多 1024 字节）
                    recv_data = client_conn.recv(1024)
                    if not recv_data:
                        print(f"❌ 设备 {client_addr} 断开连接\n")
                        break

                    # 打印设备数据
                    print(f"📥 接收设备数据: {recv_data.decode().strip()}")

                    # 向设备发送回复（可选，根据需求修改）
                    reply_data = (
                        f"电脑已收到你的消息: {recv_data.decode().strip()}".encode()
                    )
                    client_conn.send(reply_data + b"\n")  # 加换行符，方便设备端读取
                    print(f"📤 向设备回复: {reply_data.decode()}\n")
            except socket.timeout:
                print(f"⌛ 设备 {client_addr} 超时未发数据，断开连接\n")
            except Exception as e:
                print(f"❌ 与设备 {client_addr} 通信异常: {e}\n")
            finally:
                client_conn.close()  # 关闭与该设备的连接
    except KeyboardInterrupt:
        print("\n🛑 服务器正在关闭...")
    finally:
        server_socket.close()
        print("✅ 服务器已关闭")


if __name__ == "__main__":
    pc_tcp_server()

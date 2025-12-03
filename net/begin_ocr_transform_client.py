import socket
import time


def main():
    # K230的IP地址
    K230_IP = "192.168.41.134"
    K230_PORT = 8888

    # 创建socket连接
    s = socket.socket()
    s.settimeout(10)

    try:
        print(f"连接到K230服务器 {K230_IP}:{K230_PORT}...")
        s.connect((K230_IP, K230_PORT))
        print("成功连接到K230！等待接收OCR识别数据...\n")

        while True:
            # 接收数据
            data = s.recv(1024)
            if not data:
                print("连接已关闭")
                break

            # 解码并显示数据
            text = data.decode("utf-8")
            print("=" * 50)
            print(text)
            print("=" * 50)

    except socket.timeout:
        print("连接超时")
    except ConnectionRefusedError:
        print("无法连接到K230，请确保服务器已启动")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        s.close()
        print("\n客户端已关闭")


if __name__ == "__main__":
    main()

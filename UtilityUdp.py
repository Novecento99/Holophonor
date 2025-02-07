import socket
import numpy as np


class UDPReceiver:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.ip, self.port))

    def receive_data(self, buffer_size):
        try:
            data, _ = self.socket.recvfrom(buffer_size)
            return data
        except Exception as e:
            print(f"Failed to receive data: {e}")
            return None

    def close(self):
        self.socket.close()


class UDPSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_data(self, data):
        try:
            self.socket.sendto(data, (self.ip, self.port))
        except Exception as e:
            print(f"Failed to send data: {e}")

    def close(self):
        self.socket.close()


if __name__ == "__main__":
    from time import sleep

    udp_sender = UDPSender("127.0.0.1", 5005)
    data = np.zeros(100)
    while True:
        for i in range(100):
            for j in range(100):
                data[i] = data[i] + 0.01 * 10
                print(i, data[i])
                udp_sender.send_data(data.tobytes())
                sleep(0.01)
            for j in range(100):
                data[i] = data[i] - 0.01 * 10
                print(i, data[i])
                udp_sender.send_data(data.tobytes())
                sleep(0.01)

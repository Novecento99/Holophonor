import torch
import numpy as np
import threading
import time
from pytorch_pretrained_biggan import BigGAN
import cv2
import UtilityUdp

# 773
# 947
# 980


# python 3.12
# media feature pack for windows N
class LiuNet_biggan:
    def __init__(self):
        self.device = "cuda"

        self.notes = torch.zeros(1, 128).to(device=self.device)
        self.current_class = torch.zeros(1, 1000).to(device=self.device)
        self.current_image = None

        self.current_class[0, 773] = 1

        self.net_gen = BigGAN.from_pretrained("biggan-deep-256")
        print(self.net_gen.config)

        self.net_gen = self.net_gen.to(device=self.device)
        self.net_gen = self.net_gen.eval()

        self.udp_receiver = UtilityUdp.UDPReceiver("127.0.0.1", 5005)

        self.window_dimension = 800

        self.performance_times_gen = []
        self.performance_times_udp = []
        self.performance_times_shw = []

    def real_time_from_udp(self):

        while True:
            data = self.udp_receiver.receive_data(1000)
            start = time.time()
            if data:
                array = np.frombuffer(data, dtype=np.float64)
                if len(array) < 5:
                    print("new subject: ", array)
                    index = int(round(array[0]))
                    classe_ = torch.zeros(1, 1000)
                    classe_[0, index] = 1
                    self.current_class = classe_
                else:

                    if len(array) < 128:
                        array = np.concatenate([array, np.zeros(128 - len(array))])
                    torch_array = (
                        torch.tensor(array).unsqueeze(0).to(device=self.device)
                    )
                    torch_array = torch_array.type(torch.FloatTensor)
                    self.notes = torch_array.clone().to(device=self.device)
                    self.performance_times_udp.append(1000 * (time.time() - start))

    def real_time_image_gen(self):

        with torch.no_grad():
            last_notes = torch.randn(1, 128).to(device=self.device)
            while True:
                while torch.equal(self.notes, last_notes) and torch.equal(
                    self.current_class, last_class
                ):
                    pass
                start = time.time()
                last_notes = self.notes.clone()
                last_class = self.current_class.clone()
                last_notes2 = last_notes.to(device=self.device)
                last_class2 = last_class.to(device=self.device)
                image = self.net_gen(last_notes2, last_class2, 1)
                image = image.squeeze(0)
                image = image.detach().cpu().numpy()
                image = (image + 1) / 2
                image = image.transpose(1, 2, 0)
                self.current_image = image
                self.performance_times_gen.append(1000 * (time.time() - start))

    def real_time_main(self):
        cv2.namedWindow("Holophonor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "Holophonor", self.window_dimension, self.window_dimension
        )  # Set initial window size

        udp_notes = threading.Thread(target=self.real_time_from_udp)
        udp_notes.daemon = True
        udp_notes.start()

        genImg_thread = threading.Thread(target=self.real_time_image_gen)
        genImg_thread.daemon = True
        genImg_thread.start()

        while True:
            while self.current_image is None:
                time.sleep(0.001)  # CANNOT PUT PASS!?
            start = time.time()
            image = self.current_image
            self.current_image = None

            start = time.time()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(
                image,
                (self.window_dimension, self.window_dimension),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imshow("Holophonor", image)
            cv2.waitKey(5)
            self.performance_times_shw.append(1000 * (time.time() - start))

            # trim performance arrays to 100 elements
            if len(self.performance_times_gen) > 100:
                self.performance_times_gen = self.performance_times_gen[-100:]
            if len(self.performance_times_udp) > 100:
                self.performance_times_gen = self.performance_times_gen[-100:]
            if len(self.performance_times_shw) > 100:
                self.performance_times_gen = self.performance_times_gen[-100:]
            try:
                gen_time = round(
                    sum(self.performance_times_gen) / len(self.performance_times_gen)
                )
                udp_time = round(
                    sum(self.performance_times_udp) / len(self.performance_times_udp)
                )
                shw_time = round(
                    sum(self.performance_times_shw) / len(self.performance_times_shw)
                )
                tot_time = gen_time + udp_time + shw_time
                max_time = max(gen_time, udp_time, shw_time)

                print(
                    f"Gen: {gen_time} ms, UDP: {udp_time} ms, Shw: {shw_time} ms, Tot: {tot_time} ms, Fps: {1000 / max_time:.1f} fps"
                )
            except:
                pass


if __name__ == "__main__":
    Holophonor_instance = LiuNet_biggan()
    Holophonor_instance.real_time_main()

import torch
import numpy as np
import threading
import time
import cv2
import UtilityUdp
from custom_model import Generator


# python 3.12
# media feature pack for windows N
class LiuNet:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # sometimes cpu is better though!
        print(f"Using device: {self.device}")
        self.notes = torch.zeros(1, 128).to(device=self.device)
        self.current_image = None

        self.udp_receiver = UtilityUdp.UDPReceiver("127.0.0.1", 5005)

        self.net_gen = Generator().to(device=self.device)
        # Load the saved model state dictionary
        self.net_gen.load_state_dict(
            torch.load(
                r"models\flowers_256p_100d_64b_400e\generator_flowers_256p_100d_64b_400e.pth",
                weights_only=True,
                map_location=torch.device(self.device),
            ),
        )
        self.net_gen.eval()
        print("Generator model loaded")

        self.window_dimension = 512

        self.performance_times_gen = []
        self.performance_times_udp = []
        self.performance_times_shw = []

    def real_time_from_udp(self):

        while True:
            data = self.udp_receiver.receive_data(1000)
            start = time.time()
            if data:
                array = np.frombuffer(data, dtype=np.float64)
                if len(array) < 128:
                    array = np.concatenate([array, np.zeros(128 - len(array))])
                torch_array = torch.tensor(array).unsqueeze(0).to(device=self.device)
                torch_array = torch_array.type(torch.FloatTensor)
                self.notes = torch_array.clone().to(device=self.device)

                # performance measurement
                self.performance_times_udp.append(1000 * (time.time() - start))

    def real_time_image_gen(self):
        with torch.no_grad():
            last_notes = torch.randn(1, 128).to(device=self.device)
            while True:
                while torch.equal(self.notes, last_notes):
                    pass
                start = time.time()
                last_notes = self.notes.clone()

                image = self.net_gen(last_notes[:, :100, None, None])
                image = image.squeeze(0)
                image = image.detach().cpu().numpy()
                image = (image + 1) / 2
                image = image.transpose(1, 2, 0)
                self.current_image = image
                self.performance_times_gen.append(1000 * (time.time() - start))

    def real_time_main(self):
        cv2.namedWindow("LiuNet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "LiuNet", self.window_dimension, self.window_dimension
        )  # Set initial window size

        udp_notes = threading.Thread(target=self.real_time_from_udp)
        udp_notes.daemon = True
        udp_notes.start()

        genImg_thread = threading.Thread(target=self.real_time_image_gen)
        genImg_thread.daemon = True
        genImg_thread.start()

        while True:
            while self.current_image is None:
                time.sleep(
                    0.001
                )  # inserting a pass here seems to deteriorate generation thread performance
            start = time.time()
            image = self.current_image
            self.current_image = None

            start = time.time()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # upscale image using cv2
            image = cv2.resize(
                image,
                (self.window_dimension, self.window_dimension),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imshow("LiuNet", image)
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
                    f"Gen: {gen_time} ms, UDP: {udp_time} ms, Shw: {shw_time} ms. Latency: {tot_time} ms, Fps: {1000 / max_time:.1f} fps"
                )
            except:
                pass


if __name__ == "__main__":
    aLiuNet = LiuNet()
    aLiuNet.real_time_main()

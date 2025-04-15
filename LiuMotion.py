print("Holophonor is importing torch...")
import torch

print("Holophonor is importing cv2...")
import cv2

print("Holophonor is importing LiuNets...")
import LiuNets
import threading
import time

import UtilityUdp
import sys

print("Modules imported...")
# 773
# 947
# 980


# python 3.12
# media feature pack for windows N
class Holophonor:
    """
    Holophonor class handles the main functionality of the application, including
    receiving UDP data, generating images using neural networks, and displaying
    the images in a window.
    """

    def __init__(self, LiuGan, udpReceiver, pixel_dimension, fullscreen=False):
        """
        Initialize the Holophonor class.

        Args:
            LiuGan: The neural network model for generating images.
            udpReceiver: The UDP receiver for receiving data.
            pixel_dimension: The dimension of the window in pixels.
            fullscreen: Boolean indicating whether to display in fullscreen mode.
        """
        self.LiuGan = LiuGan
        self.udp_receiver = udpReceiver

        self.window_dimension = pixel_dimension
        self.current_image = None

        self.gen_delay = []
        self.udp_delay = []
        self.shw_delay = []
        self.fullscreen = fullscreen

    def udp_listener(self):
        """
        Listen for UDP data and process it using the neural network model.
        """
        print("UDP listener started...")
        while True:
            data = self.udp_receiver.receive_data(16000)
            start = time.time()
            if data:
                self.LiuGan.digest_input(data)

                self.udp_delay.append(1000 * (time.time() - start))

    def neural_use(self):
        """
        Generate images using the neural network model.
        """
        print("Image generation started...")
        with torch.no_grad():
            while True:
                start = time.time()
                self.current_image = self.LiuGan.generate_output()
                self.gen_delay.append(1000 * (time.time() - start))

    def main(self):
        """
        Main loop for displaying the generated images in a window.
        """
        if self.fullscreen:
            cv2.namedWindow("Holophonor", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "Holophonor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.namedWindow("Holophonor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "Holophonor", self.window_dimension, self.window_dimension
            )  # Set initial window size

        udp_notes = threading.Thread(target=self.udp_listener)
        udp_notes.daemon = True
        udp_notes.start()

        genImg_thread = threading.Thread(target=self.neural_use)
        genImg_thread.daemon = True
        genImg_thread.start()

        print("Main loop started...")
        while True:
            while self.current_image is None:
                time.sleep(
                    0.001
                )  # a pass instruction here degrades generation thread performance
            start = time.time()
            image = self.current_image
            self.current_image = None

            start = time.time()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if self.fullscreen:
                screen_height, screen_width = (
                    cv2.getWindowImageRect("Holophonor")[3],
                    cv2.getWindowImageRect("Holophonor")[2],
                )
                aspect_ratio = image.shape[1] / image.shape[0]
                new_width = int(screen_height * aspect_ratio)
                new_height = screen_height
                if new_width > screen_width:
                    new_width = screen_width
                    new_height = int(screen_width / aspect_ratio)
                resized_image = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )
                top = (screen_height - new_height) // 2
                bottom = screen_height - new_height - top
                left = (screen_width - new_width) // 2
                right = screen_width - new_width - left
                image = cv2.copyMakeBorder(
                    resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT
                )
            else:
                image = cv2.resize(
                    image,
                    (self.window_dimension, self.window_dimension),
                    interpolation=cv2.INTER_CUBIC,
                )
            cv2.imshow("Holophonor", image)
            cv2.waitKey(5)
            self.shw_delay.append(1000 * (time.time() - start))

            # trim performance arrays to 100 elements
            if len(self.gen_delay) > 100:
                self.gen_delay = self.gen_delay[-100:]
            if len(self.udp_delay) > 100:
                self.gen_delay = self.gen_delay[-100:]
            if len(self.shw_delay) > 100:
                self.gen_delay = self.gen_delay[-100:]
            try:
                gen_time = round(sum(self.gen_delay) / len(self.gen_delay))
                udp_time = round(sum(self.udp_delay) / len(self.udp_delay))
                shw_time = round(sum(self.shw_delay) / len(self.shw_delay))
                tot_time = gen_time + udp_time + shw_time
                max_time = max(gen_time, udp_time, shw_time)

                print(
                    f"Gen: {gen_time} ms, UDP: {udp_time} ms, Shw: {shw_time} ms, Tot: {tot_time} ms, Fps: {1000 / max_time:.1f} fps"
                )
            except:
                pass


if __name__ == "__main__":
    # sys.argv contains the list of command line arguments
    # sys.argv[0] is the script name, and sys.argv[1:] are the arguments passed to the script

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    if len(sys.argv) > 1:
        if sys.argv[1] == "biggan":
            LiuNet = LiuNets.LiuBigganNet(device, "biggan-deep-256")
        elif sys.argv[1] == "custom":
            LiuNet = LiuNets.LiuCustomNet(device)
        else:
            print("Invalid argument. Usage: python Holophonor.py [biggan/custom]")
            sys.exit()
    else:
        print(
            "No argument provided. Usage: python Holophonor.py [biggan/custom]. Going to use biggan model as default."
        )
        LiuNet = LiuNets.LiuBigganNet(device, "biggan-deep-256")

    print("LiuNet created...")
    udpReceiver = UtilityUdp.UDPReceiver("127.0.0.1", 5005)
    print("UDPReceiver created...")
    fullscreen = "--fullscreen" in sys.argv
    motion = Holophonor(LiuNet, udpReceiver, 800, fullscreen=False)
    print("Holophonor created...")
    motion.main()

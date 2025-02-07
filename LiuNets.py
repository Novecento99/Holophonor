from abc import ABC, abstractmethod

from pytorch_pretrained_biggan import BigGAN
import torch
import numpy as np

from LiuTrain import Generator


class LiuNet(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def digest_input(self, data):
        pass

    @abstractmethod
    def generate_output(self, input_tensor):
        pass


class LiuBigganNet(LiuNet):

    def __init__(self, device, model_name="biggan-deep-256"):  # Add 'device' parameter
        super().__init__(device)  # Call the parent class constructor
        print(model_name)
        self.noise = torch.zeros(1, 128).to(device=self.device)
        self.subject = torch.zeros(1, 1000).to(device=self.device)
        self.subject[0, 773] = 1

        self.model_name = model_name
        self.net_gen = BigGAN.from_pretrained(self.model_name)
        self.net_gen = self.net_gen.to(device=self.device)
        self.net_gen = self.net_gen.eval()
        print("Generator model loaded")

    def digest_input(self, data):
        if data:
            array = np.frombuffer(data, dtype=np.float64)
            if len(array) < 5:
                print("new subject: ", array)
                index = int(round(array[0]))
                classe_ = torch.zeros(1, 1000)
                classe_[0, index] = 1
                self.subject = classe_
            else:
                if len(array) > 128:
                    array = array[:128]
                if len(array) < 128:
                    array = np.concatenate([array, np.zeros(128 - len(array))])
                torch_array = torch.tensor(array).unsqueeze(0).to(device=self.device)
                torch_array = torch_array.type(torch.FloatTensor)
                self.noise = torch_array.clone().to(device=self.device)

    def generate_output(self):
        with torch.no_grad():
            last_noise = self.noise.clone()
            last_class = self.subject.clone()
            last_notes2 = last_noise.to(device=self.device)
            last_class2 = last_class.to(device=self.device)
            image = self.net_gen(last_notes2, last_class2, 1)
            image = image.squeeze(0)
            image = image.detach().cpu().numpy()
            image = (image + 1) / 2
            image = image.transpose(1, 2, 0)
            return image


class LiuCustomNet(LiuNet):

    def __init__(
        self,
        device,
        model_path=r"custom_models\flowers_256p_100d_64b_400e\generator_flowers_256p_100d_64b_400e.pth",
    ):  # Add 'device' parameter
        super().__init__(device)  # Call the parent class constructor
        self.noise = torch.zeros(1, 128).to(device=self.device)

        self.net_gen = Generator().to(device=self.device)
        self.model_path = model_path
        # Load the saved model state dictionary
        self.net_gen.load_state_dict(
            torch.load(
                self.model_path,
                weights_only=True,
                map_location=torch.device(self.device),
            ),
        )
        self.net_gen.eval()
        print(f"{model_path} loaded")

    def digest_input(self, data):
        array = np.frombuffer(data, dtype=np.float64)
        # if array > 128, it will be truncated
        if len(array) > 128:
            array = array[:128]
        if len(array) < 128:
            array = np.concatenate([array, np.zeros(128 - len(array))])

        torch_array = torch.tensor(array).unsqueeze(0).to(device=self.device)
        torch_array = torch_array.type(torch.FloatTensor)
        self.noise = torch_array.clone().to(device=self.device)

    def generate_output(self):
        with torch.no_grad():
            image = self.net_gen(self.noise[:, :100, None, None])
            image = image.squeeze(0)
            image = image.detach().cpu().numpy()
            image = (image + 1) / 2
            image = image.transpose(1, 2, 0)
            return image


if __name__ == "__main__":
    # test the LiuBigganNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_gen = LiuBigganNet(device, "biggan-deep-256")
    data = np.zeros(100)
    while True:
        for i in range(100):
            for j in range(100):
                data[i] = data[i] + 0.01 * 10
                print(i, data[i])
                net_gen.digest_input(data.tobytes())
                image = net_gen.generate_output()
                # plot image

# from pytorch_pretrained_biggan import BigGAN
# import torch
# model = BigGAN.from_pretrained('biggan-deep-128')
# torch.save(model.state_dict(), "./model_state_dict.pth")


import torch
import onnx
from pytorch_pretrained_biggan import BigGAN

model = BigGAN.from_pretrained("biggan-deep-128")

# Define the correct inputs for the BigGAN model
latent_vector = torch.randn(1, 128)  # Latent vector with dimension 128
class_vector = torch.zeros(
    1, 1000
)  # Class vector for 1000 classes (one-hot or softmax)
truncation = torch.tensor(0.4)  # Truncation value as a scalar tensor

# Export the model to ONNX format
torch.onnx.export(
    model,
    (latent_vector, class_vector, truncation),  # Provide all required inputs
    "biggan.onnx",
    opset_version=11,
    input_names=["latent_vector", "class_vector", "truncation"],
    output_names=["output"],
    dynamic_axes={
        "latent_vector": {0: "batch_size"},
        "class_vector": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

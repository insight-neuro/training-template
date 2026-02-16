import torch
from dotenv import load_dotenv

from .train import train

if __name__ == "__main__":
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    train()

from torch import nn
from transformers import AutoModel

from src.datamodules.unsmile_datamodule import UnsmileDataModule


class SimpleTransformerNet(nn.Module):
    def __init__(
        self,
        pretrained_path: str = None,
        input_size: int = 1024,
        hidden_size: int = 512,
        output_size: int = None,
    ):
        super().__init__()

        self.output_size = output_size
        self.bart = AutoModel.from_pretrained(pretrained_path).encoder
        self.fcn1 = nn.Linear(input_size, hidden_size)
        self.fcn2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.bart(x)["last_hidden_state"][:, 0, :]
        output = self.fcn1(output)
        output = self.relu(output)
        output = self.fcn2(output)
        output = self.sigmoid(output)
        return output

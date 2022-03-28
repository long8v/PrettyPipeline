from torch import LongTensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class UnsmileDataset(Dataset):
    def __init__(self, dataset: str = None, pretrained_model: str = None):
        self.lines = open(dataset, "r").readlines()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_len = self.tokenizer.model_max_length
        self.lineno = len(self.lines)
        self.category_num = len(self.lines[0].split("\t")[1:])

    def __getitem__(self, idx):
        line = self.lines[idx + 1]  # in oredr to skip header
        splitted_line = line.strip().split("\t")
        text = splitted_line[0]
        category = [int(ctg) for ctg in splitted_line[1:]]
        return LongTensor(
            self.tokenizer.encode(text, max_length=self.max_len, padding="max_length")
        ), LongTensor(category)

    def __len__(self):
        return self.lineno - 1


if __name__ == "__main__":
    dataset = UnsmileDataset(
        "./data/korean_unsmile_data/unsmile_train_v1.0.tsv", "facebook/mbart-large-cc25"
    )
    print(dataset[0])

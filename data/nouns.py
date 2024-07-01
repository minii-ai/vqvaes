import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


def make_datasets(
    train_size: float = 0.8,
    seed: int = 0,
    image_size: int = 128,
    normalize: bool = True,
):
    dataset = load_dataset("m1guelpf/nouns", split="train")
    splits = dataset.train_test_split(train_size=train_size, seed=seed, shuffle=True)
    train, test = splits["train"], splits["test"]
    transform = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if normalize:
        transform.append(v2.Normalize((0.5,), (0.5,)))

    transform = v2.Compose(transform)

    train_set, test_set = NounsDataset(train, transform), NounsDataset(test, transform)
    return train_set, test_set


def create_dataloader(
    nouns_dataset: "NounsDataset",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    def collate_fn(batch):
        images = [data["image"] for data in batch]
        texts = [data["text"] for data in batch]

        images = torch.stack(images, dim=0)

        return {"images": images, "texts": texts}

    return DataLoader(
        nouns_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


class NounsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data["image"] = self.transform(data["image"])

        return data

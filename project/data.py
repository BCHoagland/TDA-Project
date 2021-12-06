from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# from torchvision.utils import save_image


class Dataset:
    def __init__(self, dataset_name, batch_size):
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class('../data', transform=transforms.ToTensor(), download=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def batches(self, labels=False):
        for img, label in self.dataloader:
            img = img.view(img.size(0), -1)               #? use for feed-forward network; not for conv network
            if labels:
                yield img, label
            else:
                yield img

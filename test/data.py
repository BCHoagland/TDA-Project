from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# from torchvision.utils import save_image


class MNIST:
    def __init__(self, batch_size):
        dataset = datasets.MNIST('../data', transform=transforms.ToTensor(), download=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def batches(self, labels=False):
        for img, label in self.dataloader:
            img = img.view(img.size(0), -1)
            if labels:
                yield img, label
            else:
                yield img

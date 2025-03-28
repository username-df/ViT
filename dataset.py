from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomResizedCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

train_dataset = datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)

train_data = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=512, shuffle=False)
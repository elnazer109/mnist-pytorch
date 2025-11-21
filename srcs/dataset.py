from torchvision.datasets import MNIST
from torchvision.transforms import Compose , transforms
from torch.utils.data import DataLoader


# load data and perform data augumentation 
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((32, 32)),

    ])

training_data= MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data= MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
    )

# DataLoader
train_dataloader = DataLoader(training_data, batch_size=128 , shuffle=True , drop_last =True)

test_dataloader = DataLoader(test_data, batch_size=128 , shuffle=True , drop_last = True )



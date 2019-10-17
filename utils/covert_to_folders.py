import torch as ch
from torchvision import transforms

train_transform = transforms.Compose([...])

data_path = "robust_CIFAR"

train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
train_set = folder.TensorDataset(train_data, train_labels, transform=train_transform)

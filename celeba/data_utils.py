from torchvision import transforms
import os
import numpy as np
from torch.utils.data import DataLoader
import utils
from PIL import Image


BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/"


class CelebACustomBinary(utils.Dataset):
    def __init__(self, root_dir, shuffle=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get filenames
        path_0, path_1 = os.path.join(
            self.root_dir, "0"), os.path.join(self.root_dir, "1")
        filenames_0 = [os.path.join(path_0, x) for x in os.listdir(path_0)]
        filenames_1 = [os.path.join(path_1, x) for x in os.listdir(path_1)]
        self.filenames = filenames_0 + filenames_1
        if shuffle:
            np.random.shuffle(self.filenames)

        self.attr_names = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
            'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
            'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
            'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
            'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young'
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        x = Image.open(filename)
        y = os.path.basename(os.path.normpath(filename)).split("_")[0]
        y = np.array([int(c) for c in y])
        if self.transform:
            x = self.transform(x)
        return x, y


class CelebaWrapper:
    def __init__(self, prop, split, features=None):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.train_dir = os.path.join(
            BASE_DATA_DIR, "/%s/%s/train" % (prop, split))
        self.test_dir = os.path.join(
            BASE_DATA_DIR, "/%s/%s/test" % (prop, split))

        self.ds_train = CelebACustomBinary(
            self.train_dir, transform=data_transform)
        self.ds_val = CelebACustomBinary(
            self.test_dir, transform=data_transform)

        # Extract attribute names
        self.attr_names = self.ds_val.attr_names

    def get_loaders(self, batch_size, shuffle=False):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        test_loader = DataLoader(
            self.ds_val, batch_size=batch_size * 2,
            shuffle=shuffle, num_workers=2)

        return train_loader, test_loader

from torchvision import transforms, models
import torch.nn as nn
import torch as ch
import pandas as pd
from torch.utils.data import DataLoader
import data_utils


def collect_features(loader, model):
    all_features = []
    for data in loader:
        x, y, _ = data
        x = x.cuda()

        features = model(x).cpu()
        all_features.append(features)

    return ch.cat(all_features, 0)


if __name__ == "__main__":
    # Load model
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Get rid of existing classification layer
    # Extract only features
    model.classifier = nn.Identity()
    model = model.cuda()
    model = nn.DataParallel(model)

    # Fetch indices for which features are wanted
    split = 2
    og_train = pd.read_csv("./data/split_%d/train.csv" % split)
    og_val = pd.read_csv("./data/split_%d/val.csv" % split)

    # Define data transforms
    input_size = 224
    data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Ready dataset objects
    ds_train = data_utils.HamDataset(og_train, data_transform)
    ds_val = data_utils.HamDataset(og_val, data_transform)

    # Ready loaders
    batch_size = 100 * 4
    train_loader = DataLoader(
        ds_train, batch_size=batch_size,
        shuffle=False, num_workers=8)
    val_loader = DataLoader(
        ds_val, batch_size=batch_size * 2,
        shuffle=False, num_workers=8)

    # Collect features
    print("Collecting train-data features")
    train_features = collect_features(train_loader, model)
    print("Collecting val-data features")
    val_features = collect_features(val_loader, model)

    # Create mapping between filepaths and features for those images
    train_map = {og_train['path'][i]: train_features[i]
                 for i in range(len(og_train))}

    val_map = {og_val['path'][i]: val_features[i]
               for i in range(len(og_val))}

    # Save features
    ch.save(train_map, "./data/split_%d/features_train.pt" % split)
    ch.save(val_map, "./data/split_%d/features_val.pt" % split)

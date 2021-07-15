from model_utils import get_pre_processor
import torch.nn as nn
import os
import torch as ch
from data_utils import BoneWrapper, get_df, BASE_DATA_DIR


def collect_features(loader, model):
    all_features = []
    for data in loader:
        x, y, _ = data
        x = x.cuda()

        features = model(x).cpu()
        all_features.append(features)

    return ch.cat(all_features, 0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=["victim", "adv"])
    args = parser.parse_args()

    # Load model
    model = get_pre_processor()
    model = model.cuda()
    model = nn.DataParallel(model)

    # Fetch indices for which features are wanted
    og_train, og_val = get_df(args.split)

    # Ready dataset objects
    wrapper = BoneWrapper(og_train, og_val)

    # Ready loaders
    batch_size = 100
    train_loader, val_loader = wrapper.get_loaders(100, shuffle=False)

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
    ft_path = os.path.join(BASE_DATA_DIR, args.split, "features_train.pt")
    fv_path = os.path.join(BASE_DATA_DIR, args.split, "features_val.pt")
    ch.save(train_map, ft_path)
    ch.save(val_map, fv_path)

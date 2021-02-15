import data_utils


if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/"
    input_size = 224
    # batch_size = 128 * 4
    batch_size = 100 * 4

    # Define augmentation
    model = data_utils.make_feature_extractor()

    # Process datasets and get features
    ds = data_utils.HamWrapper(
        "./data/split_1/original/train.csv",
        "./data/split_1/original/test.csv")
    train_loader, val_loader = ds.get_loaders(batch_size)

    data_utils.train(model, (train_loader, val_loader), lr=1e-3, epoch_num=20)

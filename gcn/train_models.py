import data_utils


if __name__ == "__main__":
    ds = data_utils.PubMedData()
    # ds = data_utils.AmazonPhotoData()

    m = data_utils.get_model(ds)
    acc = data_utils.train_model(ds, 0.5, m, 2e-3, 1000)

    print('Accuracy: {:.4f}'.format(acc))

    data = ds.get_features()
    y = ds.get_labels()

    out = m(data)
    data_utils.visualize(out, color=y.cpu())

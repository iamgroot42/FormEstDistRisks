import numpy as np
import torch as ch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import utils
import os
from sklearn.neural_network import MLPClassifier
from facenet_pytorch import MTCNN


def get_cropped_faces(cropmodel, x):
    x_cropped = []
    for x_ in x:
        x_ = (x_ * 0.5) + 0.5
        image = Image.fromarray(
            (255 * np.transpose(x_.numpy(), (1, 2, 0))).astype('uint8'))
        try:
            x_cr = cropmodel(image)
            x_cropped.append(x_cr)
        except Exception:
            continue
    return ch.stack(x_cropped, 0)


if __name__ == "__main__":

    constants = utils.Celeb()
    ds = constants.get_dataset()

    cropmodel = MTCNN(device='cuda')

    folders = [
        "/u/as9rw/work/fnb/implems/celeba_models/"
        "smile_old_vggface_cropped_augs/",
        "/u/as9rw/work/fnb/implems/celeba_models/"
        "smile_all_vggface_cropped_augs/",
        "/u/as9rw/work/fnb/implems/celeba_models/"
        "smile_attractive_vggface_cropped_augs/",
        "/u/as9rw/work/fnb/implems/celeba_models/"
        "smile_male_vggface_cropped_augs/"
    ]

    _, dataloader = ds.make_loaders(
        batch_size=100, workers=8, shuffle_val=True, only_val=True)
    # Sample random points from dataloader
    x, _ = next(iter(dataloader))
    x = get_cropped_faces(cropmodel, x)

    X, Y = [], []
    for j, f in tqdm(enumerate(folders)):
        for i, path in enumerate(os.listdir(f)):
            # Take 10 classifiers per folder
            if i == 10:
                break

            model = utils.FaceModel(512, train_feat=True).cuda()
            model = nn.DataParallel(model)
            model.load_state_dict(ch.load(os.path.join(f, path)))
            model.eval()

            features = model(x.cuda(), only_latent=True).detach()
            X.append(features.cpu().numpy())
            Y.append(j)

    X = np.array(X)
    Y = np.array(Y)

    X_, Y_ = [], []
    for i in range(Y.shape[0]):
        if Y[i] == 3:
            Y_.append(0)
            X_.append(X[i].flatten())
        elif Y[i] == 1:
            Y_.append(1)
            X_.append(X[i].flatten())

    X_, Y_ = np.array(X_), np.array(Y_)
    clf = MLPClassifier(hidden_layer_sizes=(1000, 100))
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X_, Y_):
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = Y_[train_index], Y_[test_index]
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))
        print()

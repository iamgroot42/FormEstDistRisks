import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from tensorflow import keras
import os

import utils


def train_as_they_said(model, trainloader, testloader, loss_fn,
                       acc_fn, base_save_path, epochs=40):
    # optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for e in range(epochs):
        # Train
        running_loss, running_acc = 0.0, 0.0
        num_samples = 0
        model.train()
        iterator = tqdm(trainloader)
        for (x, y) in iterator:
            x, y = x.cuda(), y.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)[:, 0]
            loss = loss_fn(outputs, y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.shape[0]
            running_acc += acc_fn(outputs, y)
            num_samples += x.shape[0]

            iterator.set_description("Epoch %d : [Train] Loss: %.5f Accuacy: %.2f" % (
                e, running_loss / num_samples, 100 * running_acc / num_samples))

        # Validation
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        num_samples = 0
        with ch.no_grad():
            for (x, y) in testloader:
                x, y = x.cuda(), y.cuda()

                outputs = model(x)[:, 0]
                loss = loss_fn(outputs, y.float())
                running_loss += loss.item() * x.shape[0]
                running_acc += acc_fn(outputs, y)
                num_samples += x.shape[0]

        print("[Val] Loss: %.5f Accuacy: %.2f\n" %
              (running_loss / num_samples, 100 * running_acc / num_samples))
        ch.save(model.state_dict(), os.path.join(base_save_path, str(
            e+1) + "_" + str(running_acc.item() / num_samples)) + ".pth")

        # scheduler.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='none',
                        help='which dataset to work on (census/mnist/celeba/processed)')
    parser.add_argument('--which', type=str, default='', help='patho to data')
    parser.add_argument('--savepath', type=str, default='',
                        help='folder where trained model(s) should be saved')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train model for')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    parser.add_argument('--weightinit', type=str, default='vggface2',
                        help='which weight initialization to use: vggface2, casia-webface, or none')
    parser.add_argument('--augment', type=bool, default=False,
                        help='use data augmentations when training models?')
    parser.add_argument('--hidden', type=str, default="64,16",
                        help='comma-separated dimensions for hidden layers for models classification layer')
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.dataset == 'census':
        # Census Income dataset
        ci = utils.CensusIncome("./census_data/")

        def sex_filter(df): return utils.filter(
            df, lambda x: x['sex:Female'] == 1, 0.65)

        def race_filter(df): return utils.filter(
            df, lambda x: x['race:White'] == 0,  1.0)

        def income_filter(df): return utils.filter(
            df, lambda x: x['income'] == 1, 0.5)

        num_cfs = 100
        for i in range(1, num_cfs + 1):
            # (x_tr, y_tr), (x_te, y_te), _ = ci.load_data()
            (x_tr, y_tr), (x_te, y_te), _ = ci.load_data(income_filter)
            # clf = RandomForestClassifier(max_depth=30, random_state=0, n_jobs=-1)
            clf = MLPClassifier(hidden_layer_sizes=(60, 30, 30), max_iter=200)
            clf.fit(x_tr, y_tr.ravel())
            print("Classifier %d : Train acc %.2f , Test acc %.2f" % (i,
                100 * clf.score(x_tr, y_tr.ravel()),
                100 * clf.score(x_te, y_te.ravel())))

            dump(clf, os.path.join('census_models_mlp_many/income/', str(i)))

    elif args.dataset == 'celeba':
        # CelebA dataset
        hidden_layer_sizes = [int(x) for x in args.hidden.split(",")]
        model = utils.FaceModel(512,
                                weight_init=args.weightinit,
                                train_feat=True,
                                hidden=hidden_layer_sizes).cuda()
        model = nn.DataParallel(model)

        path = args.which

        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        train_transforms = test_transforms[:]
        if args.augment:
            augment_transforms = [
                transforms.RandomAffine(degrees=20,
                                        translate=(0.2, 0.2),
                                        shear=0.2),
                transforms.RandomHorizontalFlip()
                ]
            train_transforms = augment_transforms + train_transforms

        train_transform = transforms.Compose(train_transforms)
        test_transform = transforms.Compose(test_transforms)

        train_set = torchvision.datasets.ImageFolder(path + "/train",
                                                     transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(path + "/test",
                                                    transform=test_transform)
        trainloader = ch.utils.data.DataLoader(train_set,
                                               batch_size=args.bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8)
        testloader = ch.utils.data.DataLoader(test_set,
                                              batch_size=args.bs,
                                              shuffle=True,
                                              num_workers=8)

        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        def acc_fn(outputs, y): return ch.sum((y == (outputs >= 0)))

        train_as_they_said(model, trainloader,
                           testloader, loss_fn,
                           acc_fn, args.savepath,
                           epochs=args.epochs)

    elif args.dataset == 'mnist':
        # MNIST
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
        x_tr = x_tr.astype("float32") / 255
        x_te = x_te.astype("float32") / 255

        x_tr = x_tr.reshape(x_tr.shape[0], -1)
        x_te = x_te.reshape(x_te.shape[0], -1)

        # Brightness Jitter
        brightness = np.random.uniform(0.1, 0.5, size=(x_tr.shape[0],))
        x_tr = x_tr + np.expand_dims(brightness, -1)
        x_tr = np.clip(x_tr, 0, 1)

        clf = MLPClassifier(hidden_layer_sizes=(128, 32, 16), max_iter=40)
        clf.fit(x_tr, y_tr)
        print(clf.score(x_tr, y_tr))
        print(clf.score(x_te, y_te))

        dump(clf, 'mnist_models/brightness_3')
    else:
        raise ValueError("Dataset not supported yet")

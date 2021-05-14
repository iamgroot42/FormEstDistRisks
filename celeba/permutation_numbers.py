import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms
import os

import utils


def train_as_they_said(model, trainloader, testloader, loss_fn,
                       acc_fn, base_save_path, lr=1e-4, epochs=15):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=str, default='', help='path to data')
    parser.add_argument('--savepath', type=str, default='',
                        help='folder where trained model(s) should be saved')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train model for')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weightinit', type=str, default='vggface2',
                        help='which weight initialization to use: vggface2, " \
                            "casia-webface, or none')
    parser.add_argument('--augment', type=bool, default=False,
                        help='use data augmentations when training models?')
    parser.add_argument('--hidden', type=str, default="64,16",
                        help='comma-separated dimensions for hidden layers " \
                        "for models classification layer')
    args = parser.parse_args()
    utils.flash_utils(args)

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
    trainloader = DataLoader(train_set,
                             batch_size=args.bs,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=8)
    testloader = DataLoader(test_set,
                            batch_size=args.bs,
                            shuffle=True,
                            num_workers=8)

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    def acc_fn(outputs, y): return ch.sum((y == (outputs >= 0)))

    train_as_they_said(model, trainloader,
                       testloader, loss_fn,
                       acc_fn, args.savepath,
                       lr=args.lr,
                       epochs=args.epochs)

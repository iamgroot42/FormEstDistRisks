import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import os
import utils


def unlearn(model, trainloader, testloader, loss_fn, acc_fn, epochs=40):
    # optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    for e in range(epochs):
        # Train
        running_loss, running_accs = 0.0, [0.0, 0.0]
        num_samples = 0
        num_keep, num_nokeep = 0, 0
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

            running_loss += loss.item() * y.shape[0]
            acc_1, acc_2, n_keep = acc_fn(outputs, y)
            running_accs[0] += acc_1
            running_accs[1] += acc_2
            num_keep += n_keep
            num_nokeep += (y.shape[0] - n_keep)
            num_samples += y.shape[0]

            iterator.set_description("Epoch %d : [Train] Loss: %.5f | "
                                     "Accuracy on D_r: %.2f | "
                                     "Accuracy on D_f %.2f" % (e,
                                                               running_loss / num_samples,
                                                               100 * running_accs[0] / num_keep,
                                                               100 * running_accs[1] / num_nokeep))

        # Validation
        model.eval()
        running_loss, running_accs = 0.0, [0.0, 0.0]
        num_samples = 0
        num_keep, num_nokeep = 0, 0
        with ch.no_grad():
            for (x, y) in testloader:
                x, y = x.cuda(), y.cuda()

                outputs = model(x)[:, 0]
                loss = loss_fn(outputs, y.float())

                running_loss += loss.item() * y.shape[0]
                acc_1, acc_2, n_keep = acc_fn(outputs, y)
                running_accs[0] += acc_1
                running_accs[1] += acc_2
                num_keep += n_keep
                num_nokeep += (y.shape[0] - n_keep)
                num_samples += y.shape[0]

            print("Epoch %d : [Validation] Loss: %.5f | "
                  "Accuracy on D_r: %.2f | "
                  "Accuracy on D_f %.2f" % (e,
                                            running_loss / num_samples,
                                            100 *
                                            running_accs[0] / num_keep,
                                            100 * running_accs[1] / num_nokeep))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=str, help='path to data')
    parser.add_argument('--modelpath', type=str, help='path to model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train model for')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    args = parser.parse_args()
    utils.flash_utils(args)

    # CelebA dataset
    model = utils.FaceModel(512, train_feat=True, weight_init=None).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(args.modelpath), strict=False)
    path = args.which

    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
    train_transforms = test_transforms[:]

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    train_set = utils.CelebACustomBinary(
        os.path.join(path, "train"),
        transform=train_transform)
    test_set = utils.CelebACustomBinary(
        os.path.join(path, "test"),
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

    loss_obj = nn.BCEWithLogitsLoss(reduction='none')

    attrs = constants.attr_names
    prop = attrs.index("Male")
    target_prop = attrs.index("Smiling")
    reg_lambda = 1e-1

    def loss_fn(outputs, y):
        keep = (y[:, prop] == 1).cpu().numpy()
        nokeep = np.logical_not(keep)
        # Retain accuracy on D_r
        loss_dr = loss_obj(outputs[keep], y[keep, target_prop])
        # Lose accuracy on D_f
        # Option 1: Make model flip its predictions
        # loss_df = - loss_obj(outputs[nokeep], y[nokeep, target_prop])
        # Option 2: Make model perform as bad as random
        random_labels = ch.randint(0, 2, (np.sum(nokeep),)).float().cuda()
        loss_df = loss_obj(outputs[nokeep], random_labels)

        combined_loss = ch.mean(loss_dr) + \
            ch.mean(ch.exp(reg_lambda * loss_df))
        return combined_loss

    def acc_fn(outputs, y):
        keep = (y[:, prop] == 1).cpu().numpy()
        nokeep = np.logical_not(keep)
        # Compute accuracy on D_r
        D_r = ch.sum((y[keep, target_prop] == (outputs[keep] >= 0)))
        D_f = ch.sum((y[nokeep, target_prop] == (outputs[nokeep] >= 0)))
        return D_r.detach().cpu().item(), \
            D_f.detach().cpu().item(), np.sum(keep)

    unlearn(model, trainloader,
            testloader, loss_fn,
            acc_fn, epochs=args.epochs)

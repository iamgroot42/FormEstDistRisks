from cleverhans.future.torch.attacks import projected_gradient_descent
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch as ch
import numpy as np
import torch.nn as nn
import utils
import implem_utils

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def collect_augmented_data(loader, deg):
    X, X_aug, Y = [], [], []
    for x, y in loader:
        X_aug.append(implem_utils.augmentation_robustness(x, deg))
        X.append(x)
        Y.append(y)
    return (X, X_aug, Y)


def get_robustness_shifts(model_fn, augdata, target, prop_id):
    counts = [0, 0]
    noprop, prop = [0, 0], [0, 0]
    for x, x_adv, y in zip(*augdata):
        y_picked = y[:, target].cuda()
        y_t = y[:, prop_id].numpy()
        # x_adv = implem_utils.augmentation_robustness(x).cuda()

        prop_idex = np.nonzero(y_t == 1)[0]
        noprop_idex = np.nonzero(y_t == 0)[0]

        before_preds = ((model_fn(x.cuda()) >= 0) == y_picked).cpu()
        after_preds = ((model_fn(x_adv) >= 0) == y_picked).cpu()

        noprop[0] += ch.sum(1.0 * before_preds[noprop_idex]).item()
        noprop[1] += ch.sum(1.0 * after_preds[noprop_idex]).item()

        prop[0] += ch.sum(1.0 * before_preds[prop_idex]).item()
        prop[1] += ch.sum(1.0 * after_preds[prop_idex]).item()

        counts[0] += prop_idex.shape[0]
        counts[1] += noprop_idex.shape[0]

    for i in range(2):
        prop[i] /= counts[0]
        noprop[i] /= counts[1]

    return noprop, prop


if __name__ == "__main__":
    eps = 10 # 30.0
    nb_iter = 200
    eps_iter = 2.5 * eps / nb_iter
    norm = 2

    paths = [
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/20_0.9235165574046058.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/augment_none/20_0.9065300896286812.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/20_0.9108834827144686.pth"
    ]

    models = []
    for MODELPATH in paths:
        model = utils.FaceModel(512,
                                train_feat=True,
                                weight_init=None).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(ch.load(MODELPATH))
        model.eval()
        models.append(model)

    # Use existing dataset instead
    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)
    dataloader = DataLoader(td,
                            batch_size=512,
                            shuffle=True)

    attrs = constants.attr_names
    target_prop = attrs.index("Smiling")
    # Look at examples that satisfy particular property
    inspect_these = ["Attractive", "Male", "Young"]

    def saveimg(x_, path):
        x_ = (x_ * 0.5) + 0.5
        image = Image.fromarray((255 * np.transpose(x_.numpy(), (1, 2, 0))).astype('uint8'))
        image.save(path)

    degrees = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    noprop_scores, prop_scores = [], []
    for deg in tqdm(degrees):
        # Collecte augmented data
        augdata = collect_augmented_data(dataloader, deg)
        npz, pz = [], []
        for j, model in enumerate(models):
            model_fn = lambda z: model(z)[:, 0]
            noprop, prop = get_robustness_shifts(model_fn,
                                                 augdata,
                                                 target_prop,
                                                 attrs.index(inspect_these[1]))

            npz.append(noprop)
            pz.append(prop)

        noprop_scores.append(npz)
        prop_scores.append(pz)
        # print("[P=0] Before and after augmentation: %.2f, %.2f" % (noprop[0], noprop[1]))
        # print("[P=1] Before and after augmentation: %.2f, %.2f" % (prop[0], prop[1]))
        # print()

    diffs = []
    for x, y in zip(prop_scores, noprop_scores):
        diff = (y[0][0] - y[0][1]) - (x[0][0] - x[0][1])
        diffs.append(diff)
    plt.plot(degrees,
             diffs,
             marker='o',
             label='all model')

    diffs = []
    for x, y in zip(prop_scores, noprop_scores):
        diff = (y[1][0] - y[1][1]) - (x[1][0] - x[1][1])
        diffs.append(diff)
    plt.plot(degrees,
             diffs,
             marker='x',
             label='male model')

    plt.legend()
    plt.savefig("../visualize/rotation_trends.png")
    exit(0)

    x_advs = []
    for x, y in dataloader:
        # Get cropped versions
        y_picked = y[:, target_prop].cuda()

        # Pick only the ones that satisfy property
        satisfy = ch.nonzero(y[:, attrs.index(inspect_these[1])])[:, 0]
        x = x[satisfy]
        y_picked = y_picked[satisfy]

        for j, model in tqdm(enumerate(models)):
            model_fn = lambda z: model(z)[:, 0]
            # y_pseudo = 1. * (model(x)[:, 0] >= 0)
            # x_adv = projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm,
            #                    clip_min=-1, clip_max=1, y=y_pseudo,
            #                    rand_init=True, sanity_checks=True,
            #                    loss_fn=nn.BCEWithLogitsLoss()).detach().cpu()
            x_adv = implem_utils.augmentation_robustness(x).cpu()
            x_advs.append(x_adv)

            print("Accuracy on normal:", ch.mean(1.0 * ((model_fn(x.cuda())>=0) == y_picked)).item())
            print("Accuracy on augmented:", ch.mean(1.0 * ((model_fn(x_adv.cuda())>=0) == y_picked)).item())
            # selected_index = 7
            # saveimg(x[selected_index].cpu(), "../visualize/normal.png")
            # saveimg(x_adv[selected_index].cpu(), "../visualize/perturbed_" + str(j) + ".png")
            # print("Saved")
            # print(model(x_)[selected_index])
            # print(model(x_adv)[selected_index])
        # exit(0)
        break
    exit(0)

    # Look at inter-model transfer for adversarial examples
    names = ["all", "all", "male", "male"]
    for i in range(len(x_advs)):
        preds_og = models[i](x_advs[i])[:, 0]
        # print("Original error on %s : %.2f" % (names[i], 1 - ch.mean(1. * (y_picked == (preds_og >=0)))))
        for j in range(i, len(x_advs)):
            preds_target = models[j](x_advs[i])[:, 0]
            print("Transfer rate to %s : %.2f" % (names[j], 1 - 1 *ch.mean(1. * (y_picked == (preds_target >=0)))))
        print()

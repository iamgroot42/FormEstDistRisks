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


if __name__ == "__main__":
    eps = 10
    nb_iter = 200
    eps_iter = 2.5 * eps / nb_iter
    norm = 2
    batch_size = 900

    paths = [
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/10_0.9233484619263742.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/20_0.9108834827144686.pth",
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
                            batch_size=batch_size,
                            shuffle=False)

    attrs = constants.attr_names
    target_prop = attrs.index("Smiling")
    # Look at examples that satisfy particular property
    inspect_these = ["Attractive", "Male", "Young"]

    def saveimg(x_, path):
        x_ = (x_ * 0.5) + 0.5
        image = Image.fromarray((255 * np.transpose(x_.numpy(), (1, 2, 0))).astype('uint8'))
        image.save(path)

    degrees = [20, 30, 40, 50, 60, 70, 80]
    jitter_vals = [0.5, 1, 2, 3, 4, 5, 6]
    translate_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    erase_vals = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    noprop_scores, prop_scores = [], []
    # for deg in tqdm(degrees):
    # for jv in tqdm(jitter_vals):
    # for tv in tqdm(translate_vals):
    for ev in tqdm(erase_vals):
        # Collecte augmented data
        # augdata = implem_utils.collect_augmented_data(dataloader,
                                                    #   translate=(tv, tv))
        # augdata = implem_utils.collect_augmented_data(dataloader, deg=deg)
        # augdata = implem_utils.collect_augmented_data(dataloader,
        #                                               jitter=(0, 0, 0, jv))
                                                    #   jitter=(jv, jv, jv, jv))
        # augdata = implem_utils.collect_augmented_data(dataloader,
        #                                               translate=(0, tv))
        augdata = implem_utils.collect_augmented_data(dataloader,
                                                      erase_scale=(ev-0.01, ev))
        # saveimg(augdata[1][0][0], "../visualize/gauss_val_%f.png" % jv)
        npz, pz = [], []
        for j, model in enumerate(models):

            def model_fn(z):
                return model(z)[:, 0]

            noprop, prop = implem_utils.get_robustness_shifts(model_fn,
                                                              augdata,
                                                              target_prop,
                                                              attrs.index(inspect_these[1]))

            npz.append(noprop)
            pz.append(prop)

        noprop_scores.append(npz)
        prop_scores.append(pz)

    diffs = []
    for x, y in zip(prop_scores, noprop_scores):
        # diff = 100 * (((y[0][0] - y[0][1]) / y[0][0]) - ((x[0][0] - x[0][1]) / x[0][0]))
        diff = (y[0][0] - y[0][1]) / (x[0][0] - x[0][1])
        diffs.append(diff)
    plt.plot(erase_vals,
             diffs,
             marker='o',
             label='more males model')

    diffs = []
    for x, y in zip(prop_scores, noprop_scores):
        # diff = 100 * (((y[1][0] - y[1][1]) / y[1][0]) - ((x[1][0] - x[1][1]) / x[1][0]))
        diff = (y[1][0] - y[1][1]) / (x[1][0] - x[1][1])
        diffs.append(diff)
    plt.plot(erase_vals,
             diffs,
             marker='x',
             label='more females model')

    plt.xlabel("Parameter for data augmentation")
    plt.ylabel("Drop in performance (%) for P=1 - Drop in performance (%) for P=0")
    plt.legend()
    # plt.savefig("../visualize/rotation_trends.png")
    # plt.savefig("../visualize/jitter_trends.png")
    # plt.savefig("../visualize/translate_trends.png")
    plt.savefig("../visualize/erase_trends.png")
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

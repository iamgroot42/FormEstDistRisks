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
    batch_size = 900

    paths = [
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/13_0.9164565473188772.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/17_0.9122541603630863.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/20_0.9108834827144686.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/13_0.900640204865557.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/17_0.9106274007682459.pth",
    ]

    # Use existing dataset instead
    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)
    dataloader = DataLoader(td,
                            batch_size=batch_size,
                            shuffle=True)

    attrs = constants.attr_names
    focus_attr = attrs.index("Male")
    target_attr = attrs.index("Smiling")
    focus_value = 1

    distrs = []
    for MODELPATH in paths:

        model = utils.FaceModel(512,
                                train_feat=True,
                                weight_init=None).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(ch.load(MODELPATH))
        model.eval()

        acts = []
        for x, y in tqdm(dataloader):
            # wanted = ch.logical_and(y[:, focus_attr] == focus_value,
            #                         y[:, target_attr] == 1)
            wanted = y[:, focus_attr] == focus_value
            latent = model(x[wanted].cuda(), only_latent=True).detach()
            act = ch.sum(latent > 0, 1).cpu().numpy()
            acts.append(act)

        distrs.append(np.concatenate(acts))

    # for distr in distrs:
    #     print(np.mean(distr))

    # Normalize around zero
    # distrs[0] -= 258
    # distrs[1] -= 257

    # distrs[0] -= 258
    # distrs[1] -= 260
    # distrs[2] -= 257
    # distrs[3] -= 258

    labels = [
        "45% males",
        "45% males",
        "45% males",
        "67% males",
        "67% males",
        "67% males"
    ]

    for i, dist in enumerate(distrs):
        plt.hist(dist,
                 bins=50,
                 density=True,
                 alpha=0.75,
                 label=labels[i])

    plt.legend()
    plt.ylim(0, 0.9)
    plt.title("Activation trends on males")
    plt.savefig("../visualize/act_distr_male.png")

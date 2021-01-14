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
    batch_size = 250 * 4
    # batch_size = 250 * 4
    # batch_size = 500 * 3
    # batch_size = 500 * 4

    paths = [
        # First ratio category
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/1/"
        "13_0.9135180520570949.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/casia/1/"
        "15_0.9125104953820319.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/vggface/1/"
        "15_0.9222502099076406.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/2/"
        "15_0.9177162048698573.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/3/"
        "15_0.9136859781696054.pth",
        # Second ratio category
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/1/"
        "15_0.9122836498067551.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/casia/1/"
        "15_0.9107712989413544.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/vggface/1/"
        "15_0.9121156108217107.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/2/"
        "15_0.9122836498067551.pth",
        "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/3/"
        "12_0.9137960006721559.pth",
    ]

    MODELPATH = paths[3]

    # Use existing dataset instead
    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/"
        "50_50/all/split_2/test",
        transform=transform)
    dataloader = DataLoader(td, batch_size=batch_size, shuffle=True)

    attrs = constants.attr_names
    focus_attr = attrs.index("Male")
    target_attr = attrs.index("Smiling")
    focus_value = 1

    distrs = []
    model = utils.FaceModel(512,
                            train_feat=True,
                            weight_init=None).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(MODELPATH), strict=False)
    model.eval()

    acts_yes, acts_no = [], []
    for x, y in tqdm(dataloader):
        prop_yes = (y[:, focus_attr] == 1)
        prop_no = (y[:, focus_attr] == 0)

        latent_yes = model(x[prop_yes].cuda(), only_latent=True,
                           deep_latent=9,
                           within_block=None).detach()
        latent_no = model(x[prop_no].cuda(), only_latent=True,
                          deep_latent=9,
                          within_block=None).detach()

        latent_yes = latent_yes.view(latent_yes.shape[0], -1)
        latent_no = latent_no.view(latent_no.shape[0], -1)
        num_dims = latent_yes.shape[1]

        act_yes = ch.sum(latent_yes > 0, 1).cpu().numpy()
        act_no = ch.sum(latent_no > 0, 1).cpu().numpy()

        acts_yes.append(act_yes)
        acts_no.append(act_no)

    # Data-agnostic normalizaion to [0, 1]
    distrs.append(np.concatenate(acts_yes) / num_dims)
    distrs.append(np.concatenate(acts_no) / num_dims)

    labels = ["Males", "Females"]
    colors = ['C0', 'C1', ]
    hatches = ['o', 'x']

    for i, dist in enumerate(distrs):
        plt.hist(dist,
                 bins=50,
                 density=True,
                 alpha=0.75,
                 label=labels[i],
                 color=colors[i],
                 hatch=hatches[i])

    plt.legend()
    plt.title("Activation trends on male (block 9)")
    plt.xlabel("Number of neurons activated")
    plt.ylabel("Normalized frequency of samples")
    plt.savefig("../visualize/act_distr_male.png")

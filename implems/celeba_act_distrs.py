from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch as ch
import numpy as np
import torch.nn as nn
import utils
import implem_utils
import itertools


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    batch_size = 250 * 3
    # batch_size = 500 * 3

    paths = [
        # First ratio category
        [
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/1/"
            "13_0.9135180520570949.pth",
            # "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/casia/1/"
            # "15_0.9125104953820319.pth",
            # "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/vggface/1/"
            # "15_0.9222502099076406.pth",
            # "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/2/"
            # "15_0.9177162048698573.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/3/"
            "15_0.9136859781696054.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/2/"
            "15_0.9177162048698573.pth",
        ],
        # Second ratio category
        [
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/1/"
            "15_0.9122836498067551.pth",
            # "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/casia/1/"
            # "15_0.9107712989413544.pth",
            # "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/vggface/1/"
            # "15_0.9121156108217107.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/2/"
            "15_0.9122836498067551.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/3/"
            "12_0.9137960006721559.pth",
        ]
    ]

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

    colors = ['C0', 'C1']
    hatches = ['*', 'o', 'x']
    labels = [[]] * len(paths)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for i in range(len(paths)):
        labels[i] = [r'$M^{%d}_{%d}$' % (j, i) for j in range(len(paths[i]))]
    for j, mtypes in enumerate(paths):
        distrs = []
        for MODELPATH in mtypes:

            model = utils.FaceModel(512,
                                    train_feat=True,
                                    weight_init=None).cuda()
            model = nn.DataParallel(model)
            model.load_state_dict(ch.load(MODELPATH), strict=False)
            model.eval()

            all_acts, acts = [], []
            for x, y in tqdm(dataloader):
                # wanted = ch.logical_and(y[:, focus_attr] == focus_value,
                #                         y[:, target_attr] == 0)
                wanted = y[:, focus_attr] == focus_value
                # latent = model(x[wanted].cuda(), only_latent=True).detach()
                # latent = model(x[wanted].cuda(), only_latent=True,
                latent = model(x.cuda(), only_latent=True).detach()
                # latent = model(x.cuda(), only_latent=True,
                #                deep_latent=3,
                #                within_block=None).detach()
                latent = latent.view(latent.shape[0], -1)
                num_dims = latent.shape[1]
                act = ch.sum(latent > 0, 1).cpu().numpy()
                # acts.append(act)
                acts.append(act[wanted])
                all_acts.append(act)

            all_acts = np.concatenate(all_acts)
            # Mean-centering distribution values per model
            median_count = int(np.median(all_acts))
            # distrs.append(np.concatenate(acts) - median_count)
            # [0-1] based on minimum-maximum activations across all data
            minc, maxc = np.min(all_acts), np.max(all_acts)
            # distrs.append((np.concatenate(acts) - minc) / (maxc - minc))
            mean, std = np.mean(all_acts), np.std(all_acts)
            # distrs.append((np.concatenate(acts) - mean) / std)

            # Data-agnostic normalizaion to [0, 1]
            distrs.append(np.concatenate(acts) / num_dims)
            print("median:", median_count)
            print("min, max:", minc, maxc)
            print("mean, std:", mean, std)
            print()

        for i, dist in enumerate(distrs):

            yspacing = 1
            hist, bin_edges = np.histogram(dist, bins=50, density=True)
            dx = np.diff(bin_edges)
            dy = np.ones_like(hist)
            y = (j * len(labels[j]) + i) * (1 + yspacing) * np.ones_like(hist)
            z = np.zeros_like(hist)

            # ax.bar3d(bin_edges[:-1], y, z, dx, dy, hist, color=colors[j],
            #          zsort='average', alpha=0.75, label=labels[j][i])

            plt.hist(dist,
                     bins=50,
                     density=True,
                     alpha=0.6,
                     label=labels[j][i],
                     color=colors[j],
                     hatch=hatches[i])

    plt.legend()
    plt.title(r"Activation distributions on $x\mid f(x)=1$")
    plt.xlabel("Fraction of neurons activated")
    plt.ylabel("Normalized frequency of samples")

    # ax.set_xlabel("Fraction of neurons activated")
    # ax.set_ylabel(r"Models $M_i^j$")
    # ax.set_zlabel("Normalized frequency of samples")
    # ax.set_yticklabels(list(itertools.chain(*labels)))

    plt.savefig("../visualize/act_distr_male.png")

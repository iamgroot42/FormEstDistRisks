from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch as ch
import numpy as np
import torch.nn as nn
import utils

from model_utils import get_model

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def layerwise_progression(model, data, indices_x, indices_y):
    activations_x, activations_y = [], []
    x = data.clone()

    blocks = model.feature_model.get_layers()

    for i, block in enumerate(blocks):
        if isinstance(block, nn.Sequential):
            block_ = block
        else:
            block_ = [block]

        for layer in block_:
            x = layer(x)

            # Flatten data
            x_ = x.view(x.shape[0], -1).clone().detach().cpu()
            # Take not of dimension-size
            num_dims = x_.shape[1]
            # Look at activated neurons (scaled to [0, 1])
            act_x = ch.sum(x_[indices_x] > 0, 1).numpy() / num_dims
            act_y = ch.sum(x_[indices_y] > 0, 1).numpy() / num_dims
            # Take note of means, mean differences
            activations_x.append(act_x)
            activations_y.append(act_y)

    activations_x = np.array(activations_x)
    activations_y = np.array(activations_y)
    return activations_x, activations_y


def calculate_area_overlap(dist1, dist2):
    bins = np.linspace(0, 1, 1000)
    h1, _ = np.histogram(dist1, bins=bins, density=True)
    h2, _ = np.histogram(dist2, bins=bins, density=True)

    bins = np.diff(bins)
    sm = 0
    for i in range(len(bins)):
        sm += bins[i] * min(h1[i], h2[i])
    return sm


def get_trends_for_model(model, dataloader):
    acts_yes, acts_no = [], []
    for x, y in tqdm(dataloader):
        prop_yes = (y[:, focus_attr] == 1)
        prop_no = (y[:, focus_attr] == 0)

        # Get all layer block activations
        act_yes, act_no = layerwise_progression(
            model.module, x.cuda(), prop_yes, prop_no)
        acts_yes.append(act_yes)
        acts_no.append(act_no)

    acts_yes = np.concatenate(acts_yes, 1)
    acts_no = np.concatenate(acts_no, 1)

    diffs = []
    for i in range(acts_yes.shape[0]):
        diffs.append(calculate_area_overlap(acts_yes[i], acts_no[i]))

    diffs = np.array(diffs)
    print("Layer-stats:", np.min(diffs), np.max(diffs), np.mean(diffs))
    print(np.argmin(diffs))
    return diffs

    # Normalize with min/max number of activations per layer
    # acts_all = np.concatenate((acts_yes, acts_no), 1)
    # acts_min = np.min(acts_all, 1, keepdims=True)
    # acts_max = np.max(acts_all, 1, keepdims=True)
    # print(acts_min[7:10], acts_max[7:10])
    # acts_yes = (acts_yes - acts_min) / (acts_max - acts_min)
    # acts_no = (acts_no - acts_min) / (acts_max - acts_min)

    # Get histogram-based area of overlap, plot those values instead
    return calculate_area_overlap(acts_yes, acts_no)

    # Get mean, stds per layer
    mean_yes = np.mean(acts_yes, 1)
    mean_no = np.mean(acts_no, 1)

    return mean_yes - mean_no


if __name__ == "__main__":
    batch_size = 150
    # batch_size = 100 * 3

    paths = [
        # First ratio category
        [
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/1/"
            "15_0.8884970612930311.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/1/"
            "11_0.9141897565071369.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/2/"
            "15_0.9177162048698573.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/2/"
            "11_0.8753988245172124.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/3/"
            "15_0.9136859781696054.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/all/none/3/"
            "11_0.9158690176322418.pth",
        ],
        # Second ratio category
        [
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/1/"
            "15_0.9122836498067551.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/1/"
            "11_0.9075785582255084.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/2/"
            "15_0.9122836498067551.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/2/"
            "11_0.8926230885565452.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/3/"
            "15_0.9000168038985045.pth",
            "/p/adversarialml/as9rw/celeb_models/50_50/split_1/male/none/3/"
            "11_0.9074105192404638.pth"
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

    colors = ['C0', 'C1']
    for i, sub_paths in enumerate(paths):
        for j, MODELPATH in enumerate(sub_paths):

            model = get_model(MODELPATH, use_prefix=False).cuda()

            y_ = get_trends_for_model(model, dataloader)
            # y_ = np.abs(y_)
            x_ = np.arange(y_.shape[0])

            # if j == 1:
                # plt.plot(x_, y_, color=colors[i], marker='o')
            # elif j == 2:
                # plt.plot(x_, y_, color=colors[i], marker='d')
            # else:
                # plt.plot(x_, y_, color=colors[i], marker='x')
            plt.plot(x_, y_, color=colors[i], marker='x')

    c0_patch = mpatches.Patch(
        color='C0', label=r'Model traind on $D\sim G_0(\mathcal{D})$')
    c1_patch = mpatches.Patch(
        color='C1', label=r'Model traind on $D\sim G_1(\mathcal{D})$')
    plt.legend(handles=[c0_patch, c1_patch])
    plt.xticks(np.arange(min(x_), max(x_)+1, 4.0))
    plt.xlabel("Model layers")
    plt.ylabel(
        r'Overlap (area) in activation distributions between $y_d=0, y_d=1$')
    plt.savefig("../visualize/intra_layers_all.png")
    exit(0)

    # MODELPATH = paths[0][3]
    pi = (0, 0)

    MODELPATH = paths[pi[0]][pi[1]]

    picked_layers = [
        [17, 11, 10, 27, 29],
        [23, 11, 24, 12, 18]
    ]

    distrs = []
    model = utils.FaceModel(512,
                            train_feat=True,
                            weight_init=None).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(MODELPATH), strict=False)
    model.eval()

    picked_layer = picked_layers[pi[0]][pi[1]]

    acts_yes, acts_no = [], []
    for x, y in tqdm(dataloader):
        prop_yes = (y[:, focus_attr] == 1)
        prop_no = (y[:, focus_attr] == 0)

        # Get all layer block activations
        if picked_layer == -1:
            act_yes, act_no = layerwise_progression(
                model.module, x.cuda(), prop_yes, prop_no)
            acts_yes.append(act_yes)
            acts_no.append(act_no)
        # Get specified layer output
        else:
            latent_yes = model(x[prop_yes].cuda(), only_latent=True,
                               deep_latent=picked_layer,
                               flatmode=True).detach()
            latent_no = model(x[prop_no].cuda(), only_latent=True,
                              deep_latent=picked_layer,
                              flatmode=True).detach()

            latent_yes = latent_yes.view(latent_yes.shape[0], -1)
            latent_no = latent_no.view(latent_no.shape[0], -1)
            num_dims = latent_yes.shape[1]

            act_yes = ch.sum(latent_yes > 0, 1).cpu().numpy()
            act_no = ch.sum(latent_no > 0, 1).cpu().numpy()

            acts_yes.append(act_yes)
            acts_no.append(act_no)

    if picked_layer == -1:
        acts_yes = np.concatenate(acts_yes, 1)
        acts_no = np.concatenate(acts_no, 1)

        # Normalize with min/max number of activations per layer
        acts_all = np.concatenate((acts_yes, acts_no), 1)
        acts_min = np.min(acts_all, 1, keepdims=True)
        acts_max = np.max(acts_all, 1, keepdims=True)
        acts_yes = (acts_yes - acts_min) / (acts_max - acts_min)
        acts_no = (acts_no - acts_min) / (acts_max - acts_min)

        # Get mean, stds per layer
        mean_yes, std_yes = np.mean(acts_yes, 1), np.std(acts_yes, 1)
        mean_no, std_no = np.mean(acts_no, 1), np.std(acts_no, 1)

        # Plot in the form of error plots across blocks
        # And visually see if any useful trends emerge
        layers = np.arange(mean_yes.shape[0])

        plt.plot(layers, mean_yes - mean_no)

        # plt.errorbar(layers, mean_yes, yerr=std_yes, label='P=1',
        #              ls='none', fmt='x')
        # plt.errorbar(layers + 0.2, mean_no, yerr=std_no, label='P=0',
        #              ls='none', fmt='x')
        plt.legend()
        plt.savefig("../visualize/intra_layer_inspect.png")

    else:
        # Data-agnostic normalizaion to [0, 1]
        distrs.append(np.concatenate(acts_yes) / num_dims)
        distrs.append(np.concatenate(acts_no) / num_dims)

        labels = ["Males", "Females"]
        colors = ['C0', 'C1', ]
        hatches = ['o', 'x']

        # Calculate overlap area
        oa = calculate_area_overlap(distrs[0], distrs[1])
        print("Overlap area: %.4f" % oa)

        for i, dist in enumerate(distrs):
            bins = np.linspace(0, 1, 1000)
            plt.hist(dist,
                     bins=50,
                     density=True,
                     alpha=0.75,
                     label=labels[i],
                     color=colors[i],
                     hatch=hatches[i])

        plt.legend()
        plt.title("Activation trends on male (block %d)" % picked_layer)
        plt.xlabel("Number of neurons activated")
        plt.ylabel("Normalized frequency of samples")
        plt.savefig("../visualize/celeba_within_act_distr.png")

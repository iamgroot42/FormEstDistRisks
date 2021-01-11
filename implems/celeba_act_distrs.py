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
    batch_size = 250 * 3
    # batch_size = 500 * 3
    # batch_size = 500 * 4

    paths = [
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
            "all/64_16/none/17_0.9122541603630863.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
            "all/64_16/augment_none/20_0.9235165574046058.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
            "all/64_16/augment_vggface/18_0.9253656076651539.pth",
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
        #     "male/64_16/none/20_0.9108834827144686.pth",
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
        #     "male/64_16/augment_none/18_0.9185659411011524.pth",
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
        #     "male/64_16/augment_vggface/17_0.9236875800256082.pth"
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
            "attractive/64_16/none/13_0.9201197053406999.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
            "attractive/64_16/augment_none/20_0.925414364640884.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/"
            "attractive/64_16/augment_vggface/20_0.9307090239410681.pth"
    ]

    # Use existing dataset instead
    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/"
        "70_30/all/split_2/test",
        transform=transform)
    dataloader = DataLoader(td, batch_size=batch_size, shuffle=True)

    attrs = constants.attr_names
    focus_attr = attrs.index("Attractive")
    target_attr = attrs.index("Smiling")
    focus_value = 1

    distrs = []
    for MODELPATH in paths:

        model = utils.FaceModel(512,
                                train_feat=True,
                                weight_init=None).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(ch.load(MODELPATH), strict=False)
        model.eval()

        all_acts, acts = [], []
        for x, y in tqdm(dataloader):
            # wanted = ch.logical_and(y[:, focus_attr] == focus_value,
            #                         y[:, target_attr] == 1)
            wanted = y[:, focus_attr] == focus_value
            # latent = model(x[wanted].cuda(), only_latent=True).detach()
            # latent = model(x[wanted].cuda(), only_latent=True,
            latent = model(x.cuda(), only_latent=True,
                           deep_latent=10).detach()
            latent = latent.view(latent.shape[0], -1)
            act = ch.sum(latent > 0, 1).cpu().numpy()
            # acts.append(act)
            acts.append(act[wanted])
            all_acts.append(act)

        distrs.append(np.concatenate(acts))
        all_acts = np.concatenate(all_acts)
        # Mean-centering distribution values per model
        median_count = int(np.median(all_acts))
        # distrs.append(np.concatenate(acts) - median_count)
        # [0-1] based on minimum-maximum activations across all data
        minc, maxc = np.min(all_acts), np.max(all_acts)
        # distrs.append((np.concatenate(acts) - minc) / (maxc - minc))
        mean, std = np.mean(all_acts), np.std(all_acts)
        # distrs.append((np.concatenate(acts) - mean) / std)
        print("median:", median_count)
        print("min, max:", minc, maxc)
        print("mean, std:", mean, std)
        print()

    labels = [
        "45% males",
        "45% males",
        "45% males",
        "67% males",
        "67% males",
        "67% males"
    ]

    colors = [
        'C0', 'C0', 'C0',
        'C1', 'C1', 'C1'
    ]

    hatches = [
        'o', '\\', 'x',
        'o', '\\', 'x'
    ]

    for i, dist in enumerate(distrs):
        plt.hist(dist,
                 bins=50,
                 density=True,
                 alpha=0.75,
                 label=labels[i],
                 color=colors[i],
                 hatch=hatches[i])

    plt.legend()
    plt.title("Activation trends on attractive (block 10)")
    plt.xlabel("Number of neurons activated")
    plt.ylabel("Normalized frequency of samples")
    plt.savefig("../visualize/act_distr_male.png")

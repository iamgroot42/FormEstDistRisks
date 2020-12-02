import numpy as np
import torch as ch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import utils
import implem_utils


def get_features_for_model(dataloader, MODELPATH, weight_init,
                           method_type, layers=[64, 16]):
    # Load model
    model = utils.FaceModel(512, train_feat=True,
                            weight_init=weight_init, hidden=layers).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(MODELPATH), strict=False)
    model.eval()

    # Get latent representations
    lat, sta = implem_utils.get_latents(model, dataloader, method_type)
    # lat = np.sort(lat, 1)
    # lat = np.array([np.std(lat, 1), np.mean(lat == 0, 1), np.mean(lat, 1), np.mean(lat ** 2, 1)]).T
    return (lat, sta)


if __name__ == "__main__":
    import argparse
    methods = ['latent', 'weightpermute', 'query', 'querysigmoid', 'permute']

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--numruns', type=int, default=10, help='number of runs (samplings) for meta-classifier')
    parser.add_argument('--sample', type=int, default=2500, help='number of query points to sample')
    parser.add_argument('--all', type=bool, default=False, help='use all checkpoints (per model) for analysis?')
    parser.add_argument('--method', type=str, default='latent', help='which method to use (%s)' % "/".join(methods))
    parser.add_argument('--mlp_tr', type=float, default=0.33, help='test ratio for meta-classifier')
    args = parser.parse_args()
    utils.flash_utils(args)

    batch_size = args.bs
    try:
        method_type = methods.index(args.method)
    except ValueError:
        print("Method %s not implemented yet: Pick one of: %s" % (args.method, "/".join(methods)))
        exit(0)

    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    folder_paths = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/augment_none/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/",
        ]
    ]

    blind_test_models = [
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/10_0.928498243559719.pth",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/10_0.9093969555035128.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/15_0.9073793914943687.pth",

        ],
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/10_0.9240681998413958.pth",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/10_0.8992862807295797.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/20_0.8947974217311234.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/15_0.9120626151012892.pth",
        ]
    ]

    # if 1 == 2:
    #     cropmodel = MTCNN(device='cuda')

    #     # Get all cropped images
    #     x_cropped, y_cropped = [], []
    #     _, dataloader = ds.make_loaders(
    #         batch_size=batch_size, workers=8, shuffle_train=False, shuffle_val=False, only_val=True)
    #     for x, y in tqdm(dataloader, total=len(dataloader)):
    #         x_, indices = utils.get_cropped_faces(cropmodel, x)
    #         x_cropped.append(x_.cpu())
    #         y_cropped.append(y[indices])

    #     # Make dataloader out of this filtered data
    #     x_cropped = ch.cat(x_cropped, 0)
    #     y_cropped = ch.from_numpy(np.concatenate(y_cropped, 0))
    #     td = TensorDataset(x_cropped, y_cropped)

    # Use existing dataset instead
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    target_prop = attrs.index("Smiling")
    all_x, all_y = [], []

    for index, UPFOLDER in enumerate(folder_paths):
        model_latents = []
        model_stats = []

        for pf in UPFOLDER:
            for MODELPATHSUFFIX in tqdm(os.listdir(pf)):
                if not args.all:
                    if not("3_" in MODELPATHSUFFIX or "8_" in MODELPATHSUFFIX or "13_" in MODELPATHSUFFIX): continue
                # MODELPATH    = os.path.join(UPFOLDER, FOLDER, wanted_model)

                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)

                # Get latent representations
                latent, all_stats = get_features_for_model(
                    cropped_dataloader, MODELPATH,
                    method_type=method_type,
                    weight_init=None)
                model_stats.append(all_stats)
                model_latents.append(latent)

                all_y.append(np.ones((latent.shape[0])) * index)

        model_latents = np.array(model_latents)
        model_stats = np.array(model_stats)

        all_x.append(model_latents)

    all_x = np.concatenate(np.array(all_x), 0)
    idxs = [np.random.permutation(all_x.shape[1])[:args.sample] for i in range(args.numruns)]
    if method_type == 1 or method_type == 4:
        # Calibrate at this point
        cali, weights = implem_utils.calibration(all_x,
                                    weighted_align=(method_type == 1))
        # Calibrate all_x (except baseline first)
        for i in range(weights.shape[0]): all_x[i+1] = np.matmul(all_x[i+1], weights[i])
    clfs = []

    # If using each point independently
    if method_type in [0, 1, 4]:
        all_x = np.concatenate(all_x, 0)
        all_y = np.concatenate(all_y, 0)

    # Train 10 classifiers on random samples
    for i in range(args.numruns):
        # haha don't go brrr
        if method_type in [0, 1, 4]:
            x_tr, x_te, y_tr, y_te = train_test_split(all_x, all_y, test_size=0.4)

        # haha go brr
        if method_type == 2:
            num_each = all_x.shape[0] // 2
            x_tr, x_te, y_tr, y_te = train_test_split(all_x[:, idxs[i]], [0] * num_each + [1] * num_each, test_size=args.mlp_tr)

        clf = MLPClassifier(hidden_layer_sizes=(64, 16))
        clf.fit(x_tr, y_tr)
        print("%.2f train, %.2f test" % (clf.score(x_tr, y_tr), clf.score(x_te, y_te)))
        clfs.append(clf)

    # Test out on unseen models
    all_scores = []
    for pc in blind_test_models:
        ac = []
        for path in pc:
            cropped_dataloader = DataLoader(td, batch_size=batch_size,
                                            shuffle=False)
            latent, _ = get_features_for_model(
                cropped_dataloader, path,
                method_type=method_type,
                weight_init=None)  # "vggface2")

            if method_type == 1 or method_type == 4:
                # Calibrate latent
                _, weights = implem_utils.calibration(np.expand_dims(latent, 0),
                                         use_ref=cali,
                                         weighted_align=(method_type == 1))
                latent = np.matmul(latent, weights[0])

            if method_type in [0, 1, 4]:
                preds = [clf.predict_proba(latent[idx])[:, 1] for idx, clf in zip(idxs, clfs)]
                print("Prediction score means: ",
                      np.mean(np.mean(preds, 1)),
                      np.std(np.mean(preds, 1)),
                      np.mean(preds, 1))

            elif method_type == 2:
                preds = [clf.predict_proba(np.expand_dims(latent[idx], 0))[0, 1] for idx, clf in zip(idxs, clfs)]
                print("Prediction score means: ",
                      np.mean(preds),
                      np.std(preds),)

            preds = np.mean(preds, 0)
            ac.append(preds)
        all_scores.append(ac)
        print()

    labels = ['Models trained on $D_0$', 'Models trained on $D_1$']
    colors = [
        ['gold', 'goldenrod', 'orange', 'darkorange'],
        ['lightsteelblue', 'cornflowerblue', 'royalblue', 'mediumblue']
    ]
    for i, ac in enumerate(all_scores):
        for j, x in enumerate(ac):
            n_bins = 50
            plot_x = x
            hist, bin_edges = np.histogram(plot_x,
                                           bins=n_bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
            # plt.plot(bin_centres, hist,
            #          '-o',
            #          color=colors[i][j],
            #          label=labels[i],
            #          markersize=10)

            plt.hist(plot_x,
                     n_bins,
                     color=colors[i][j],
                     label=labels[i],
                     alpha=0.9)

    patches = []
    for i in range(len(all_scores[0])):
        patches.append(Patch(facecolor=colors[0][i], edgecolor='black'))
        patches.append(Patch(facecolor=colors[1][i], edgecolor='black'))

    # Hack to solve alignment issues when legend label includes underscore
    patches.append(Patch(facecolor='white', edgecolor='white'))
    patches.append(Patch(facecolor='white', edgecolor='white'))

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    plt.legend(handles=patches,
               labels=[''] * (2 * len(all_scores[0])) + [labels[0], labels[1]],
               ncol=len(all_scores[0])+1,
               handletextpad=0.5,
               handlelength=1.0,
               columnspacing=-0.5)

    plt.xlabel("Likelihood the model was trained on $D_1$")
    plt.ylabel("Number of datapoints")
    plt.savefig("../visualize/score_distrs_celeba.png")

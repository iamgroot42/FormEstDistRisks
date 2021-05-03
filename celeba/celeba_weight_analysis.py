import utils
import implem_utils

import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    import argparse
    methods = [
        'latent', 'weightpermute', 'query',
        'querysigmoid', 'permute', 'sortorder'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--numruns', type=int, default=10, help='number of runs (samplings) for meta-classifier')
    parser.add_argument('--sample', type=int, default=2500, help='number of query points to sample')
    parser.add_argument('--all', type=bool, default=False, help='use all checkpoints (per model) for analysis?')
    parser.add_argument('--method', type=str, default='latent', help='which method to use (%s)' % "/".join(methods))
    parser.add_argument('--mlp_tr', type=float, default=0.33, help='test ratio for meta-classifier')
    parser.add_argument('--pca', type=int, default=0, help='use PCA-based reduction?')
    parser.add_argument('--normalize_lat', type=bool, default=False, help='normalize latent-features (element-wise)?')
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

    pca_dim = args.pca
    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    common_prefix = "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/"
    folder_paths = [
        [
            common_prefix + "split_2/all/64_16/augment_none/",
            common_prefix + "split_2/all/64_16/none/",
        ],
        [
            common_prefix + "split_2/male/64_16/augment_none/",
            common_prefix + "split_2/male/64_16/none/",
        ]
    ]

    blind_test_models = [
        [
            common_prefix + "split_2/all/64_16/augment_vggface/",
            common_prefix + "split_1/all/vggface/"
        ],
        [
            common_prefix + "split_2/male/64_16/augment_vggface/",
            common_prefix + "split_1/male/vggface/"
        ]
    ]

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
            for j, MODELPATHSUFFIX in tqdm(enumerate(os.listdir(pf))):
                if not args.all and j % 3 != 0:
                    continue

                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                cropped_dataloader = DataLoader(td,
                                                batch_size=batch_size,
                                                shuffle=False)

                # Get latent representations
                latent, all_stats = implem_utils.get_features_for_model(
                    cropped_dataloader, MODELPATH,
                    method_type=method_type,
                    weight_init=None,
                    normalize_lat=args.normalize_lat)
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
        for i in range(weights.shape[0]):
            all_x[i+1] = np.matmul(all_x[i+1], weights[i])

    clfs = []

    # If using each point independently
    if method_type in [0, 1, 4, 5]:
        all_x = np.concatenate(all_x, 0)
        all_y = np.concatenate(all_y, 0)

        # Dimensionality reduction, if requested
        if pca_dim > 0:
            pca = PCA(n_components=pca_dim)
            print("Fitting PCA")
            all_x = pca.fit_transform(all_x)

    # Train 10 classifiers on random samples
    for i in range(args.numruns):
        # haha don't go brrr
        if method_type in [0, 1, 4, 5]:
            x_tr, x_te, y_tr, y_te = train_test_split(all_x, all_y, test_size=args.mlp_tr)

            # Augment training data with permutations/shuffles
            # aug_factor = 5

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
            for j, MODELPATH in enumerate(os.listdir(path)):
                # Look at only one model per test folder (for now)
                if j != 15:
                    continue
                fullpath = os.path.join(path, MODELPATH)
                cropped_dataloader = DataLoader(td, batch_size=batch_size,
                                                shuffle=False)
                latent, _ = implem_utils.get_features_for_model(
                    cropped_dataloader, fullpath,
                    method_type=method_type,
                    weight_init=None,
                    normalize_lat=args.normalize_lat)  # "vggface2")

                if method_type == 1 or method_type == 4:
                    # Calibrate latent
                    _, weights = implem_utils.calibration(np.expand_dims(latent, 0),
                                                          use_ref=cali,
                                                          weighted_align=(method_type == 1))
                    latent = np.matmul(latent, weights[0])

                    # Dimensionality reduction, if requested
                    if pca_dim > 0:
                        latent = pca.transform(latent)

                if method_type in [0, 1, 4, 5]:
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
    labels = labels[::-1]
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
            #          markersize=9)

            plt.hist(plot_x,
                     n_bins,
                     color=colors[i][j],
                     label=labels[i],
                     alpha=0.95)

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

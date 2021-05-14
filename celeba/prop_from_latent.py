import utils
import implem_utils

import numpy as np
import torch as ch
import torch.nn as nn
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
    return (lat, sta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--numruns', type=int, default=2, help='number of runs (samplings) for meta-classifier')
    parser.add_argument('--sample', type=int, default=5000, help='number of query points to sample')
    parser.add_argument('--all', type=bool, default=False, help='use all checkpoints (per model) for analysis?')
    parser.add_argument('--mlp_tr', type=float, default=0.3, help='test ratio for meta-classifier')
    parser.add_argument('--pca', type=int, default=0, help='use PCA-based reduction?')
    args = parser.parse_args()
    utils.flash_utils(args)

    batch_size = args.bs
    constants = utils.Celeb()
    ds = constants.get_dataset()

    pca_dim = args.pca
    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    folder_paths = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/male/augment_vggface/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/male/vggface/",
        ]
    ]

    # Use existing dataset instead
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    # Main idea: use latent representations to predict property-attribute
    # See if there is a difference in latent features' ability to capture
    # property, based on the ratio of that data that was present in original
    # training data
    target_prop = attrs.index("Smiling")
    for index, UPFOLDER in enumerate(folder_paths):
        print("For models of type %d" % index)
        all_scores = []
        for pf in UPFOLDER:
            for j, MODELPATHSUFFIX in enumerate(os.listdir(pf)):
                if not args.all and (j % 4 != 0):
                    continue

                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                cropped_dataloader = DataLoader(td,
                                                batch_size=batch_size,
                                                shuffle=False)

                # Get latent representations
                latent, all_stats = implem_utils.get_features_for_model(
                    cropped_dataloader, MODELPATH,
                    method_type=0,
                    weight_init=None)

                # Dimensionality reduction, if requested
                if pca_dim > 0:
                    pca = PCA(n_components=pca_dim)
                    print("Fitting PCA")
                    latent = pca.fit_transform(latent)

                # Prepare to train meta-classifier
                y_for_meta = all_stats[:, inspect_these.index("Male")]

                # Train multiple times
                scores = []
                for i in range(args.numruns):
                    idx = np.random.permutation(y_for_meta.shape[0])[:args.sample]
                    all_x, all_y = latent[idx], y_for_meta[idx]
                    x_tr, x_te, y_tr, y_te = train_test_split(all_x,
                                                              all_y,
                                                              test_size=args.mlp_tr)

                    clf = MLPClassifier(hidden_layer_sizes=(128, 64, 16, 8))
                    clf.fit(x_tr, y_tr)
                    scores.append(clf.score(x_te, y_te))
                    print("%.2f train, %.2f test" % (clf.score(x_tr, y_tr), scores[-1]))

                all_scores.append(np.mean(scores))

        plt.plot(np.arange(len(all_scores)),
                 sorted(all_scores),
                 marker='o')
        print("\n")

    plt.xlabel("Different models")
    plt.ylabel("Accuracy for property task")
    plt.savefig("../visualize/property_transfer_perf_check.png")

import utils
import numpy as np
import torch as ch
import torch.nn as nn
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_these_images(folder):
    images = []
    for file in os.listdir(folder):
        im = np.asarray(Image.open(os.path.join(folder, file))
                        ).astype(np.float32) / 255.
        images.append(im)
    images = np.array(images).transpose(0, 3, 1, 2)
    # Normalize to [-1, 1]
    images = (images - 0.5) / 0.5
    return ch.from_numpy(images)


if __name__ == "__main__":

    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    blind_test_models = [

        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/"
        "augment_vggface/20_0.9151053864168618.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/"
        "augment_vggface/10_0.928498243559719.pth",

        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/male/"
        "augment_vggface/20_0.9246347941567065.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/male/"
        "augment_vggface/10_0.9243027888446215.pth",

        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/20_0.9259516256938938.pth",
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/10_0.9240681998413958.pth",
    ]

    qp_data = get_these_images("/u/as9rw/work/fnb/visualize/dump")
    labels = ["all", "all", "attractive", "attractive"]

    sorted_order = None
    all_scores = []
    for i, MODELPATH in enumerate(blind_test_models):
        # Load model
        model = utils.FaceModel(512, train_feat=True,
                                weight_init=None, hidden=[64, 16]).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(ch.load(MODELPATH), strict=False)
        model.eval()

        preds = model(qp_data.cuda())
        preds = ch.sigmoid(preds.detach().cpu())
        x = preds.numpy()[:, 0]
        if i == 0:
            sorted_order = np.argsort(x)
        plt.plot(np.arange(x.shape[0]), x[sorted_order], label=labels[i])

        all_scores.append(x)

    plt.legend()
    plt.savefig("/u/as9rw/work/fnb/visualize/qp_scores_celeba.png")

    # Take note of differences across different kind of models
    print(np.mean(np.abs(all_scores[0] - all_scores[1])), "all-all")
    print(
        np.mean(np.abs(all_scores[2] - all_scores[3])), "attractive-attractive")
    # Print agreement
    print(np.sum((all_scores[0] >= 0.5) == (all_scores[1] >= 0.5)), "all-all")
    print(np.sum((all_scores[2] >= 0.5) == (
        all_scores[3] >= 0.5)), "attractive-attractive")
    print()
    print(np.mean(np.abs(all_scores[0] - all_scores[2])), "all-attractive")
    print(np.mean(np.abs(all_scores[0] - all_scores[3])), "all-attractive")
    print(np.mean(np.abs(all_scores[1] - all_scores[2])), "all-attractive")
    print(np.mean(np.abs(all_scores[1] - all_scores[3])), "all-attractive")
    # Print agreement
    print(np.sum((all_scores[0] >= 0.5) == (
        all_scores[2] >= 0.5)), "all-attractive")
    print(np.sum((all_scores[0] >= 0.5) == (
        all_scores[3] >= 0.5)), "all-attractive")
    print(np.sum((all_scores[1] >= 0.5) == (
        all_scores[2] >= 0.5)), "all-attractive")
    print(np.sum((all_scores[1] >= 0.5) == (
        all_scores[3] >= 0.5)), "all-attractive")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch as ch
import numpy as np
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    first_cat = " 0.5"

    # Set dark background style
    # plt.style.use('dark_background')

    data = []
    columns = [
        r'Ratio of females ($\alpha$) in dataset that model is trained on',
        "Meta-classifier accuracy (%) differentiating between models"
    ]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/0.5/")
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/0.5/")

    categories = ["0.2", "0.3", "0.4", "0.6", "0.7", "0.8"]

    # dims, vecs_train_1 = get_model_features(train_dir_1)
    # _, vecs_test_1 = get_model_features(test_dir_1)
    # vecs_train_1 = np.array(vecs_train_1)
    # vecs_test_1 = np.array(vecs_test_1)

    # for cat in categories:

    #     train_dir_2 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % cat)
    #     test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % cat)
    #     features_avail = False

    #     # Load models, convert to features
    #     _, vecs_train_2 = get_model_features(train_dir_2)
    #     _, vecs_test_2 = get_model_features(test_dir_2)
    #     vecs_train_2 = np.array(vecs_train_2)
    #     vecs_test_2 = np.array(vecs_test_2)

    #     # Ready test data
    #     Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
    #     Y_test = ch.from_numpy(np.array(Y_test)).cuda()
    #     X_test = np.concatenate((vecs_test_1, vecs_test_2))

    #     for j in range(n_tries):

    #         shuffled_1 = np.random.permutation(len(vecs_train_1))[:num_train]
    #         vecs_train_1_use = vecs_train_1[shuffled_1]

    #         shuffled_2 = np.random.permutation(len(vecs_train_2))[:num_train]
    #         vecs_train_2_use = vecs_train_2[shuffled_2]

    #         # Ready train, test data
    #         Y_train = [0.] * len(vecs_train_1_use) + \
    #             [1.] * len(vecs_train_2_use)
    #         Y_train = ch.from_numpy(np.array(Y_train)).cuda()
    #         X_train = np.concatenate((vecs_train_1_use, vecs_train_2_use))

    #         # Train meta-classifier model
    #         metamodel = PermInvModel(dims)
    #         metamodel = metamodel.cuda()

    #         _, vacc = train_meta_model(
    #                         metamodel,
    #                         (X_train, Y_train),
    #                         (X_test, Y_test),
    #                         epochs=100, binary=True,
    #                         regression=False,
    #                         lr=0.001, batch_size=batch_size,
    #                         eval_every=5)

    #         data.append([cat, vacc])

    raw_data = [
        [99.05, 99.75, 99.9, 99.9, 99.75, 99.65, 99.85, 99.8, 99.35, 100.0],
        [94.95, 93.3, 91.9, 96.4, 95.85, 96.2, 88.95, 92.95, 94.3, 96.85],
        [50.0, 51.15, 60.3, 83.35, 57.0, 79.65, 75.9, 64.95, 71.8, 69.6],
        [68.7, 70.05, 60.0, 67.6, 54.4, 60.5, 59.05, 51.75, 57.95, 57.15],
        [70.2, 90.3, 75.7, 83.75, 86.0, 89.45, 85.1, 83.7, 78.35, 72.35],
        [99.45, 99.2, 98.75, 98.4, 97.1, 92.05, 96.0, 96.75, 96.6, 97.35]
    ]
    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([categories[i], raw_data[i][j]])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(x=columns[0], y=columns[1], data=df)
    sns_plot.set(ylim=(50, 100))

    # Add vertical dashed line to signal comparison ratio
    plt.axvline(x=2.5, color='black', linewidth=1.0, linestyle='--')

    # Make sure axis label not cut off
    plt.tight_layout()

    sns_plot.figure.savefig("./meta_boxplot.png")

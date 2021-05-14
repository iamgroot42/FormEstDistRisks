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


# 0.2: 99.75, 98.9, 99.35, 99.55, 99.65
# 0.3: 91.65, 95.6, 82.9, 91.5, 91.4 
# 0.4:  RUNNING 80.6, 65.6, 50, 53.85, 61.6
# 0.6: RUNNING 83.2, 61.15, 74.05, 49.25, 62.7
# 0.7: RUNNING
# 0.8: 97.85, 97.75, 97.0, 98.4, 98.9


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

    dims, vecs_train_1 = get_model_features(train_dir_1)
    _, vecs_test_1 = get_model_features(test_dir_1)
    vecs_train_1 = np.array(vecs_train_1)
    vecs_test_1 = np.array(vecs_test_1)

    categories = ["0.2", "0.3", "0.4", "0.6", "0.7", "0.8"]
    for cat in categories:

        train_dir_2 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % cat)
        test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % cat)
        features_avail = False

        # Load models, convert to features
        _, vecs_train_2 = get_model_features(train_dir_2)
        _, vecs_test_2 = get_model_features(test_dir_2)
        vecs_train_2 = np.array(vecs_train_2)
        vecs_test_2 = np.array(vecs_test_2)

        # Ready test data
        Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
        Y_test = ch.from_numpy(np.array(Y_test)).cuda()
        X_test = np.concatenate((vecs_test_1, vecs_test_2))

        for j in range(n_tries):

            shuffled_1 = np.random.permutation(len(vecs_train_1))[:num_train]
            vecs_train_1_use = vecs_train_1[shuffled_1]

            shuffled_2 = np.random.permutation(len(vecs_train_2))[:num_train]
            vecs_train_2_use = vecs_train_2[shuffled_2]

            # Ready train, test data
            Y_train = [0.] * len(vecs_train_1_use) + \
                [1.] * len(vecs_train_2_use)
            Y_train = ch.from_numpy(np.array(Y_train)).cuda()
            X_train = np.concatenate((vecs_train_1_use, vecs_train_2_use))

            # Train meta-classifier model
            metamodel = PermInvModel(dims)
            metamodel = metamodel.cuda()

            _, vacc = train_meta_model(
                            metamodel,
                            (X_train, Y_train),
                            (X_test, Y_test),
                            epochs=100, binary=True,
                            regression=False,
                            lr=0.001, batch_size=batch_size,
                            eval_every=5)

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(x=columns[0], y=columns[1], data=df)
    sns_plot.set(ylim=(50, 100))

    # Add vertical dashed line to signal comparison ratio
    plt.axvline(x=2.5, color='black', linewidth=1.0, linestyle='--')

    # Make sure axis label not cut off
    plt.tight_layout()

    sns_plot.figure.savefig("./meta_boxplot.png")

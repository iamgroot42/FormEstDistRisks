import torch as ch
import numpy as np
import os
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model


if __name__ == "__main__":
    import sys
    first_cat = sys.argv[1]
    second_cat = sys.argv[2]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % first_cat)
    train_dir_2 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % second_cat)
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % first_cat)
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % second_cat)
    features_avail = False

    # Load models, convert to features
    dims, vecs_train_1 = get_model_features(train_dir_1)
    _, vecs_train_2 = get_model_features(train_dir_2)

    _, vecs_test_1 = get_model_features(test_dir_1)
    _, vecs_test_2 = get_model_features(test_dir_2)

    vecs_train_1 = np.array(vecs_train_1)
    vecs_train_2 = np.array(vecs_train_2)

    Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()
    X_test = vecs_test_1 + vecs_test_2
    X_test = np.array(X_test)

    accs = []
    for i in range(n_tries):

        shuffled_1 = np.random.permutation(len(vecs_train_1))[:num_train]
        vecs_train_1_use = vecs_train_1[shuffled_1]

        shuffled_2 = np.random.permutation(len(vecs_train_2))[:num_train]
        vecs_train_2_use = vecs_train_2[shuffled_2]

        # Ready train, test data
        Y_train = [0.] * len(vecs_train_1_use) + [1.] * len(vecs_train_2_use)
        Y_train = ch.from_numpy(np.array(Y_train)).cuda()
        X_train = np.concatenate((vecs_train_1_use, vecs_train_2_use))
        X_train = np.array(X_train)

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
                         eval_every=10)
        accs.append(vacc)
        print("Run %d: %.2f" % (i+1, vacc))
    
    print(accs)

import torch as ch
import numpy as np
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model


if __name__ == "__main__":
    import sys
    first_cat = sys.argv[1]
    second_cat = sys.argv[2]

    batch_size = 1000

    train_dir_1 = "%ssplit_1/%s/" % (BASE_MODELS_DIR, first_cat)
    train_dir_2 = "%ssplit_1/%s/" % (BASE_MODELS_DIR, second_cat)
    test_dir_1 = "%ssplit_2/%s/" % (BASE_MODELS_DIR, first_cat)
    test_dir_2 = "%ssplit_2/%s/" % (BASE_MODELS_DIR, second_cat)
    features_avail = False

    # Load models, convert to features
    dims, vecs_train_1 = get_model_features(train_dir_1, max_read=700)
    _, vecs_train_2 = get_model_features(train_dir_2, max_read=700)

    _, vecs_test_1 = get_model_features(test_dir_1)
    _, vecs_test_2 = get_model_features(test_dir_2)

    # Ready train, test data
    Y_train = [0.] * len(vecs_train_1) + [1.] * len(vecs_train_2)
    Y_train = ch.from_numpy(np.array(Y_train)).cuda()
    X_train = vecs_train_1 + vecs_train_2

    Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()
    X_test = vecs_test_1 + vecs_test_2

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Batch data
    print("Batching data: hold on")

    # Train meta-classifier model
    # metamodel =PermInvModel(dims, inside_dims=[128, 64, 32])
    metamodel = PermInvModel(dims)
    metamodel = metamodel.gpu()

    train_meta_model(metamodel,
                     (X_train, Y_train),
                     (X_test, Y_test),
                     epochs=200, binary=True,
                     regression=False,
                     lr=0.001, batch_size=batch_size,
                     eval_every=10)

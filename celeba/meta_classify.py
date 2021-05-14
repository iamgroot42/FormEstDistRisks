import model_utils
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import os


if __name__ == "__main__":
    common_prefix = "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/"
    folder_paths = [
        ["split_2/all/64_16/augment_none/", "split_2/all/64_16/none/"],
        ["split_2/male/64_16/augment_none/", "split_2/male/64_16/none/"]
    ]

    blind_test_models = [
        ["split_2/all/64_16/augment_vggface/", "split_1/all/vggface/"],
        ["split_2/male/64_16/augment_vggface/", "split_1/male/vggface/"]
    ]

    model_vectors = []
    labels = []
    for index, UPFOLDER in enumerate(folder_paths):
        for pf in UPFOLDER:
            for j, MODELPATHSUFFIX in tqdm(enumerate(os.listdir(pf))):
                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)

                # Load model
                model = model_utils.get_model(MODELPATH)
                vec = model_utils.extract_dl_model_weights(model)

                # Store model representation vector, label
                labels.append(index)
                model_vectors.append(vec)

    model_vectors = np.array(model_vectors)
    labels = np.array(labels)

    # Train meta-classifier
    kf = KFold(n_splits=5)
    train_scores, test_scores = [], []
    for train_index, test_index in kf.split(model_vectors):
        X_train, X_test = model_vectors[train_index], model_vectors[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = MLPClassifier(hidden_layer_sizes=(512, 128, 32, 16),
                            max_iter=500)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
    print("Train:", np.mean(train_scores), np.std(train_scores))
    print("Test:", np.mean(test_scores), np.std(test_scores))
    # Train on all data now
    clf = MLPClassifier(hidden_layer_sizes=(512, 128, 32, 16),
                        max_iter=500)
    clf.fit(model_vectors, labels)
    print("Test perf", clf.score(model_vectors, labels))

    # Test on unseen models
    model_vectors = []
    labels = []
    for index, UPFOLDER in enumerate(blind_test_models):
        for pf in UPFOLDER:
            for j, MODELPATHSUFFIX in tqdm(enumerate(os.listdir(pf))):
                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)

                # Load model
                model = model_utils.get_model(MODELPATH)
                vec = model_utils.extract_dl_model_weights(model)

                # Store model representation vector, label
                labels.append(index)
                model_vectors.append(vec)

    model_vectors = np.array(model_vectors)
    labels = np.array(labels)
    # Log performance on unseen models
    print("Accuracy on unseen models:", clf.score(model_vectors, labels))

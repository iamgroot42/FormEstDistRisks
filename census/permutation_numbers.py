from sklearn.neural_network import MLPClassifier
from joblib import dump
import os

import utils


if __name__ == "__main__":
    # Census Income dataset
    ci = utils.CensusIncome("./census_data/")

    def sex_filter(df): return utils.filter(
        df, lambda x: x['sex:Female'] == 1, 0.65)

    def race_filter(df): return utils.filter(
        df, lambda x: x['race:White'] == 0,  1.0)

    def income_filter(df): return utils.filter(
        df, lambda x: x['income'] == 1, 0.5)

    num_cfs = 100
    for i in range(1, num_cfs + 1):
        # (x_tr, y_tr), (x_te, y_te), _ = ci.load_data()
        (x_tr, y_tr), (x_te, y_te), _ = ci.load_data(income_filter)
        # clf = RandomForestClassifier(max_depth=30, random_state=0, n_jobs=-1)
        clf = MLPClassifier(hidden_layer_sizes=(60, 30, 30), max_iter=200)
        clf.fit(x_tr, y_tr.ravel())
        print("Classifier %d : Train acc %.2f , Test acc %.2f" % (i,
              100 * clf.score(x_tr, y_tr.ravel()),
              100 * clf.score(x_te, y_te.ravel())))

        dump(clf, os.path.join('census_models_mlp_many/income/', str(i)))

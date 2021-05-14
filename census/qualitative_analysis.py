import numpy as np
import os
import data_utils
from model_utils import load_model, get_models_path


if __name__ == "__main__":
    base_path = "census_models_mlp"
    paths = ['none', 'income', 'sex', 'race']
    ci = data_utils.CensusIncome()

    _, (x_te, y_te), cols = ci.load_data()
    cols = list(cols)
    desired_property = cols.index("race:White")

    # Focus on performance of desired property
    desired_ids = x_te[:, desired_property] >= 0

    need_em = []
    for path_seg in paths:
        per_model = []
        for path in os.listdir(get_models_path(path_seg, "adv", value=0.5)):
            clf = load_model(os.path.join(base_path, path_seg, path))

            preds = clf.predict(x_te)
            per_model.append(preds)
        per_model = np.array(per_model)
        avg_pred = np.mean(per_model, 0)
        need_em.append(np.around(avg_pred))

    need_em = np.array(need_em)

    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            ids = np.nonzero(need_em[i] == need_em[j])[0]
            vals, counts = np.unique(x_te[ids, desired_property],
                                     return_counts=True)
            print(vals, counts[0]/np.sum(counts))
        print()

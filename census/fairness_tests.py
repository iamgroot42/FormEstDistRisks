import utils
import data_utils
from model_utils import load_model

from tqdm import tqdm
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from aif360.datasets import BinaryLabelDataset

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def collect_data_for_models(folder_paths,
                            all_x,
                            aif_dataset,
                            plot_metric=None,
                            sample=None):
    all_metrics = []
    for index, UPFOLDER in enumerate(folder_paths):
        for pf in UPFOLDER:
            modelList = os.listdir(pf)
            if sample:
                modelList = np.random.permutation(modelList)[:sample]
            for MODELPATHSUFFIX in tqdm(modelList):
                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                # Only consider last three models per folder
                if not ("13_" in MODELPATHSUFFIX or "14_" in MODELPATHSUFFIX or "15_" in MODELPATHSUFFIX):
                    continue

                # Load model
                model = load_model(MODELPATH)

                # Get predictions
                preds = model.predict_proba(all_x)[:, 1]

                # Get model's predictions
                clasf_dataset = aif_dataset.copy(deepcopy=True)
                clasf_dataset.scores = preds

                # Reset label values as well
                thresh = 0.5
                fav_inds = clasf_dataset.scores > thresh
                clasf_dataset.labels[fav_inds] = clasf_dataset.favorable_label
                clasf_dataset.labels[~fav_inds] = clasf_dataset.unfavorable_label

                metrics = utils.compute_metrics(aif_dataset,
                                                clasf_dataset,
                                                unprivileged_groups=[
                                                    {"race": 0}],
                                                privileged_groups=[
                                                    {"race": 1}])

                all_metrics.append(metrics)

        if plot_metric is not None:
            ys = sorted([x[plot_metric] for x in all_metrics])
            xs = np.arange(len(ys))
            plt.plot(xs, ys,
                     label=str(index+1),
                     marker='o',
                     linestyle='dashed')
            plt.savefig("../visualize/metric_%s.png" % plot_metric)
            plt.clf()

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None,
                        help='number of models to sample per folder')
    parser.add_argument('--path1', type=str, default='celeba',
                        help='path to models trained on D0')
    parser.add_argument('--path2', type=str, default='celeba',
                        help='path to models trained on D1')
    parser.add_argument('--testpath1', type=str, default='celeba',
                        help='path to unseen models trained on D0')
    parser.add_argument('--testpath2', type=str, default='celeba',
                        help='path to unseen models trained on D1')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Load test data
    ci = data_utils.CensusIncome()
    _, (all_x, all_y), cols = ci.load_data(None,
                                           first=False,
                                           test_ratio=0.5)

    data = {}
    data['income'] = all_y[:, 0]
    data['sex'] = 1 * (all_x[:, cols.get_loc("sex:Female")] > 0)
    data['race'] = 1 * (all_x[:, cols.get_loc("race:White")] > 0)

    df = pd.DataFrame(data=data)
    aif_dataset = BinaryLabelDataset(df=df,
                                     label_names=["income"],
                                     protected_attribute_names=["race"])

    folder_paths = [[args.path1], [args.path2]]
    blind_test_models = [[args.testpath1], [args.testpath2]]

    # Collect all metrics for given folders
    all_metrics = collect_data_for_models(folder_paths,
                                          all_x,
                                          aif_dataset,
                                          sample=args.sample)

    # Train meta-classifier
    x_meta, y_meta = [], []
    x_meta = [list(mm.values()) for mm in all_metrics]
    y_meta = [0] * (len(x_meta) // 2) + [1] * (len(x_meta) // 2)

    clf = RandomForestClassifier(max_depth=4,
                                 n_estimators=10)  # random_state=42)\
    clf.fit(x_meta, y_meta)
    print("Meta classifier score:", clf.score(x_meta, y_meta))

    all_metrics_test = collect_data_for_models(blind_test_models,
                                               all_x,
                                               aif_dataset,
                                               sample=args.sample)
    x_meta_test = [list(mm.values()) for mm in all_metrics_test]
    y_meta_test = np.array(
        [0] * (len(x_meta_test) // 2) + [1] * (len(x_meta_test) // 2))
    scores = clf.predict_proba(x_meta_test)[:, 1]
    print(clf.score(x_meta_test, y_meta_test))

    zero_scores = sorted(scores[y_meta_test == 0])
    plt.plot(np.arange(len(zero_scores)),
             zero_scores, label="0", marker='o')
    one_scores = sorted(scores[y_meta_test == 1])
    plt.plot(np.arange(len(one_scores)), one_scores, label="1", marker='o')
    #  linestyle='dashed')

    plt.xlabel("Models (sorted by meta-classifier score)")
    plt.ylabel("Probability of model being trained on D1")
    plt.title("Fairness-based meta-classifier for Census")
    plt.savefig("../visualize/fairness_census_metaclf.png")

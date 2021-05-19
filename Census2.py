import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from tensorflow import keras
import os

import utils


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='census',
                        help='which dataset to work on (census/mnist/celeba)')
    parser.add_argument('--sexRatio', type=float,
                        default='0.5', help='ratio of females in test dataset')
    parser.add_argument('--raceRatio', type=float,
                        default='0.5', help='ratio of whites in test dataset')
    parser.add_argument('--incomeRatio', type=float,
                        default='0.5', help='ratio of income in test dataset')
    #parser.add_argument('--filterType', type=str,
                        #default='sex_filter', help='which ratio to focus on')

    args = parser.parse_args()
    utils.flash_utils(args)

    if args.dataset == 'census':
		# Census Income dataset

        base_path = "/p/adversarialml/jyc9fyf/census_models_mlp"
        paths = ['original', 'income', 'sex', 'race']
        ci = utils.CensusIncome("./census_data/")

        # Set dataset ratios
		
        sex_filter = lambda df: utils.filter(
		    df, lambda x: x['sex:Female'] == 1, args.sexRatio) #args.sexRatio
        race_filter = lambda df: utils.filter(
		    df, lambda x: x['race:White'] == 0,  args.raceRatio)
        income_filter = lambda df: utils.filter(
		    df, lambda x: x['income'] == 1, args.incomeRatio)

        #Loading data for sex filter
        _, (x_te, y_te), cols = ci.load_data()
        cols = list(cols)
		# desired_property = cols.index("sex:Female")
        desired_property = cols.index("race:White")
		# Focus on performance of desired property
		# desired_ids = (y_te == 1)[:,0]
        desired_ids = x_te[:, desired_property] >= 0
        x_te, y_te = x_te[desired_ids], y_te[desired_ids]

		# Get intermediate layer representations
        from sklearn.neural_network._base import ACTIVATIONS

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 200

        def layer_output(data, MLP, layer=0):
            L = data.copy()
            for i in range(layer):
                L = ACTIVATIONS['relu'](np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
            #print(MLP.intercepts_[0])
            return L

        cluster_them = []
        cluster_everything = []

        #inter-folder
        for path_seg in paths:
            plotem = []
            perfs = []
            for path in os.listdir(os.path.join(base_path, path_seg)): #intra-folder
                clf = load(os.path.join(base_path, path_seg, path))

                # Get output of initial layer
                z = layer_output(x_te, clf, 0)
                cluster_them.append(z)
            
                perfs.append(clf.predict_proba(x_te)[:,0])
                #print(perfs)
                for sc in clf.predict_proba(x_te)[:,0]: plotem.append(sc)
            cluster_everything.append(cluster_them) 

            #Find best image
            #print(utils.best_target_image(x_te, 0))
				# perfs.append(clf.score(x_te, y_te.ravel()))
            #bins = np.linspace(0, 1, 100)
			# plt.hist(plotem, bins, alpha=0.5, label=path_seg)
            print("%s : %.4f +- %.4f" % (path_seg, np.mean(perfs), np.std(perfs)))

#Check Failures and write in txt
base_path = "/p/adversarialml/jyc9fyf/census_models_mlp"
paths = ['original', 'income', 'sex', 'race']
failures = open(r"failures.txt", "w")

for path_seg in paths:
           
    for path in os.listdir(os.path.join(base_path, path_seg)):
        clf = load(os.path.join(base_path, path_seg, path))

        

        failCount = 0
        predict0 = 0
        predict1 = 0
        for i, j in zip(clf.predict(x_te), y_te):
    
            if(i != j):
                failCount += 1
                if(i == 1):
                    predict1 += 1
                if(i == 0):
                    predict0 += 1
                

        #failures.write("Target: " + str(j))
        #failures.write("Prediction: " + str(i))
        #failures.write("") 
        ##Print folder
        failures.write("Folder: " + path_seg + "Fails: " + str(failCount) + " Predicted 1: " + str(predict1 / failCount) + " Predicted 0: " + str(predict0 / failCount) + "\n")
        print("Folder: " + path_seg + "Fails: " + str(failCount) + " Predicted 1: " + str(predict1) + " Predicted 0: " + str(predict0) + "\n")
#print(y_te != clf.predict(x_te))

failures.close()        
"""
# Plotting
# plt.legend(loc='upper right')
		# plt.savefig("../visualize/score_dists.png")

        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        kmeans = KMeans(n_clusters=4, random_state=0)
        kmeans.fit(np.concatenate(cluster_them, 0))	

        colors  = ['indianred', 'limegreen', 'blue', 'orange']
        markers = ['o', 'x', '^', 'd']

		# For visualization
        pca = PCA(n_components=2).fit(np.concatenate(cluster_them))
        pca.fit(np.concatenate(cluster_them, 0))
        transformed = []
        for j, ct in enumerate(cluster_them):
	        np.random.shuffle(ct)
	        lab, cou = np.unique(kmeans.predict(ct), return_counts=True)
	        print(lab, cou)
	        labels = kmeans.predict(ct[:2000])
	        transformed = pca.transform(ct[:2000])
	        for x, l in zip(transformed, labels): plt.scatter(x[0], x[1], s=10, c=colors[l], marker=markers[j])
			# for x, l in zip(transformed, labels): plt.scatter(x[0], x[1], s=10, c=colors[j], marker=markers[l])

        plt.savefig("/u/jyc9fyf/fnb/visualize1/cluster_show.png")

"""
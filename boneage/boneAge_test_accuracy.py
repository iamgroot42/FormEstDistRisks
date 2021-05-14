from sklearn.model_selection import train_test_split
import numpy as np
import os
import utils
from glob import glob
import pandas as pd
import data_utils
import torch as ch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def process_data_sex(path, value, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, "split_2/val.csv"))

    # Get rid of age and sex == 0                                       ############
    wanted = (df.gender == value)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df

def process_data_age(path, value, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, "split_2/val.csv"))

    # Get rid of age and sex == 0                                       ############
    wanted = (df.age == value)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df
    
def get_stats(mainmodel, dataloader, return_preds=True):

    all_preds = [] if return_preds else None
    incorrectCount = 0
    for (x, y, (sex)) in (dataloader):

        y_ = y.cuda()

        print(x.shape)

        preds = mainmodel(x.cuda()).detach()[:, 0]
        incorrect = ((preds >= 0) != y_)
        incorrectCount = incorrect.sum().item()
        #stats.append(y[incorrect].cpu().numpy())
        if return_preds:
            all_preds.append(preds.cpu().numpy())
            #all_stats.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    total = dataloader.y.size()
    accuracy = (total - incorrectCount) / total

    return all_preds, accuracy


def run_tests(df):
    accModel1 = []
    accModel2 = []

    for FOLDER in tqdm(folder_paths):
        model_preds = []
        
        

        print(FOLDER)

        for path in os.listdir(FOLDER)[:10]:
            
            #Tests the accuracies of each model in FOLDER and returns a list of all accuracies

            MODELPATH = os.path.join(FOLDER, path)

            model = data_utils.BoneModel(1024)
            model = model.cuda()

            model.load_state_dict(ch.load(MODELPATH))
            model.eval()

            features = {}
            features["train"] = ch.load("./data/split_2/features_train.pt")
            features["val"] = ch.load("./data/split_2/features_val.pt")

            ds = data_utils.BoneWrapper(df, df, features)


            _ , test_loader = ds.get_loaders(batch_size, shuffle=False)

            print(next(iter(test_loader)))


            preds, accuracy = get_stats(model, test_loader, return_preds = True)

            if(FOLDER == folder_paths[0]):
                accModel1.append(accuracy)
            else:
                accModel2.append(accuracy)

    return accModel1, accModel2
            

if __name__ == "__main__":
    base = "/u/jyc9fyf/fnb/boneage/data/"


    
    batch_size = 250 #Any number works

    folder_paths = [
        "/p/adversarialml/as9rw/models_boneage/split_2/0.2", #test models in folders
        "/p/adversarialml/as9rw/models_boneage/split_2/0.5"
    ]
    model_name = []

    #Change data ratios and test models
    df_val = pd.read_csv("./data/split_2/val.csv")
    def filter(x): return x["gender"] == 1
    df = utils.heuristic(
        df_val, filter, .2,
        10000, class_imbalance=1.0, n_tries=300)
    accModel1, accModel2 = run_tests(df)

    df = utils.heuristic(
        df_val, filter, .5,
        10000, class_imbalance=1.0, n_tries=300)
    accModel12, accModel22 = run_tests(df)


    #Graphing

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(np.sort(accModel1), np.arange(100))
    ax1.plot(np.sort(accModel2), np.arange(100))
    ax1.plot(np.sort(accModel12), np.arange(100))
    ax1.plot(np.sort(accModel22), np.arange(100))

    plt.title("Accuracy of BoneAge Models on Gender Ratios") 


    plt.savefig("/u/jyc9fyf/bonePlots/Accuracy_05_02")


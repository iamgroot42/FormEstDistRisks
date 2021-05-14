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
    # filters dataset so that all labels for sex are a value (1 or 0)

    df = pd.read_csv(os.path.join(path, "splits/split_2/val.csv"))

    wanted = (df.sex == value)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df


    
def process_data_age(path, value, split_second_ratio=0.5):
    # filters dataset so that all labels for age are a value (1 or 0)

    df = pd.read_csv(os.path.join(path, "splits/split_2/val.csv"))

    wanted = (df.age == value)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df
    

def get_stats(mainmodel, dataloader, return_acts=True):
    # gets the number of activations in a model on a dataset

    all_acts = [] if return_acts else None
    activationCount = 0

    for (x, y, (age,sex)) in (dataloader):

        y_ = y.cuda()

        acts = mainmodel(x.cuda(), latent=0).detach()

        activationCount = ch.sum(acts > 0, 1).cpu().numpy()

        print(acts.shape)
        print(activationCount.shape)
        
        if return_acts:
            all_acts.append(activationCount)
        
        

    all_acts = np.concatenate(all_acts)

    print(all_acts)

    return all_acts


def run_tests():
    graphActivations = []

    for FOLDER in tqdm(folder_paths):    

        print(FOLDER)
        for path in os.listdir(FOLDER)[:5]:
        
            MODELPATH = os.path.join(FOLDER, path)

            #prepping model and data
            model = data_utils.HamModel(1024)
            model = model.cuda()
            model.load_state_dict(ch.load(MODELPATH))
            model.eval()

            features = ch.load(os.path.join(base, "splits/split_2/features_val.pt"))

            df1 = data_utils.HamDataset(df, features, processed=True)
            test_loader = data_utils.DataLoader(
                df1, batch_size=batch_size * 2,
                shuffle=False, num_workers=2)

            #get activations           
            activations = get_stats(model, test_loader, return_acts = True)

            
            model_name.append(FOLDER)
            graphActivations.append(activations)



    return graphActivations
            

if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/" #path to the dataset
    actsSex1 = []
    actsSex0 = []
    actsAge1 = []
    actsAge0 = []


    df = process_data_sex(base,value = 1)
    batch_size = 250 #Any number works

    folder_paths = [
        "/u/jyc9fyf/hamModels/hamAge/testAge0.2", #test models in folders
        "/u/jyc9fyf/hamModels/hamAge/testAge0.8",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.2",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.8"
    ]
    model_name = []

    actsSex1 = run_tests()

    df = process_data_sex(base,value = 0)
    actsSex0 = run_tests()

    df = process_data_age(base, value = 1)
    actsAge1 = run_tests()

    df = process_data_age(base, value = 0)
    actsAge0 = run_tests()

                   


    #Graphing

    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    #ax1.hist(actsSex1, bins = 20, alpha = .5, label = "Sex 1") #plots activations in histogram bins
    #ax1.hist(actsSex0, bins = 20, alpha = .5, label = "Sex 0") #unused subplots are commented out
    ax1.hist(actsAge1, bins = 20, alpha = .5, label = "Age 1")
    ax1.hist(actsAge0, bins = 20, alpha = .5, label = "Age 0")

    plt.title("Activations of Models in Test Folder on Age") 
    plt.legend(loc='upper right')

    plt.savefig("/u/jyc9fyf/hamPlots/test")


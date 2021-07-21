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


# filters dataset so that all labels for sex are a value (1 or 0) 
def process_data_sex(path, value, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, "splits/split_2/val.csv"))

    wanted = (df.sex == value)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df

# filters dataset so that all labels for age are a value (1 or 0)
def process_data_age(path, value, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, "splits/split_2/val.csv"))

    wanted = (df.age == value)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df
    
#Returns the number of neuron activations in the model on test data
def get_stats(mainmodel, dataloader, return_acts=True):

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


#Runs tests on models
def run_tests():

    #Used to get each set of activations from each model
    activations1 = []
    activations2 = []
    activations3 = []
    activations4 = []

    for FOLDER in tqdm(folder_paths):
        model_preds = []
        
        
            

        print(FOLDER)

        for path in os.listdir(FOLDER)[:1]:
        

            MODELPATH = os.path.join(FOLDER, path)

            model = data_utils.HamModel(1024)
            model = model.cuda()

            model.load_state_dict(ch.load(MODELPATH))
            model.eval()

            features = ch.load(os.path.join(base, "splits/split_2/features_val.pt"))


            df1 = data_utils.HamDataset(df, features, processed=True)

            test_loader = data_utils.DataLoader(
                df1, batch_size=batch_size * 2,
                shuffle=False, num_workers=2)

           
            activations = get_stats(model, test_loader, return_acts = True)

            
            model_name.append(FOLDER)

            #Separating into individual activations for each model
            if FOLDER == folder_paths[0]:
                activations1 = list(activations)  #flatten before appending
            if FOLDER == folder_paths[1]:
                activations2 = list(activations)
            if FOLDER == folder_paths[2]:
                activations3 = list(activations)
            else:
                activations4 = list(activations)



    return activations1, activations2, activations3, activations4
            

if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/"

    
    activate1 = []
    activate2 = []
    activate3 = []
    activate4 = []

    df = process_data_sex(base,value = 1) #process data before testing
    batch_size = 250 #Any number works

    folder_paths = [
        "/u/jyc9fyf/hamModels/hamAge/testAge0.2", #test models in folders
        "/u/jyc9fyf/hamModels/hamAge/testAge0.8",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.2",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.8"
    ]
    model_name = []

    activate1,activate2,activate3,activate4 = run_tests()



    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    
    #plots activations for all models
    ax1.hist(activate1, bins = 20, alpha = .5)
    ax1.hist(activate2, bins = 20, alpha = .5)
    ax1.hist(activate3, bins = 20, alpha = .5)
    ax1.hist(activate4, bins = 20, alpha = .5)

    plt.title("Activations of Models in Test Folder on Sex = 1") 
    #plt.legend(loc='upper right')

    plt.savefig("/u/jyc9fyf/hamPlots/ActivationsPlot4Sex1")


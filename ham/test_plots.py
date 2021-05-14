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



def process_data(path, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, "splits/split_2/val.csv"))

    # Get rid of age and sex == 0                                       ############
    wanted = (df.sex == 1)
    df = df[wanted]
    df = df.reset_index(drop = True)


    # Return stratified split
    return df
    
def get_stats(mainmodel, dataloader, return_preds=True):

    all_preds = [] if return_preds else None
    incorrectCount = 0
    for (x, y, (age,sex)) in (dataloader):

        y_ = y.cuda()

        #print(x.shape)

        preds = mainmodel(x.cuda()).detach()[:, 0]
        incorrect = ((preds >= 0) != y_)
        incorrectCount = incorrect.sum().item()
        #stats.append(y[incorrect].cpu().numpy())
        if return_preds:
            all_preds.append(preds.cpu().numpy())
            #all_stats.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    

    return all_preds, incorrectCount


if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/"
    allIncorrect = []

    df = process_data(base) #Assuming df_victim is unnecessary       #############
    batch_size = 250 * 8 * 4 #Any number works

    folder_paths = [
        "/u/jyc9fyf/hamModels/hamAge/testAge0.2", #test models in folders
        "/u/jyc9fyf/hamModels/hamAge/testAge0.8",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.2",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.8"
    ]

    totalIncorrect = 0 
    model_name = []

    for FOLDER in tqdm(folder_paths):
        model_preds = []
        #averageIncorrect = totalIncorrect / 10
        #print(averageIncorrect)
        totalIncorrect = 0 
        
              

        print(FOLDER)
        #wanted_model = os.listdir(FOLDER)[:10]
        #wanted_model = os.listdir(FOLDER)[:10]

        for path in os.listdir(FOLDER)[:10]:
        

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

            preds, incorrect = get_stats(model, test_loader, return_preds = True)

            allIncorrect.append(incorrect)
            #print(incorrect)
            totalIncorrect += incorrect
            model_preds.append(preds)
            model_name.append(FOLDER)
            if path == os.listdir(FOLDER)[9]:
                averageIncorrect = totalIncorrect / 10
                print(totalIncorrect)
                print(averageIncorrect)
                
    
    

    exit(0)
    
    #Graphing
    x = ["Age0.2"] #* 10
    x += ["Age0.8"] #* 10 
    
    x2 = ["Sex0.2"] #* 10 
    x2 += ["Sex0.8"] #* 10

    plt.scatter(x, allIncorrect[:2]) #Since every 10 models is a separate folder
    plt.title("Accuracy of Age Models in Test Folder") 

    plt.savefig("/u/jyc9fyf/hamPlots/ageTestPlot")

    plt.scatter(x2, allIncorrect[2:]) #replace allIncorrect with allIncorrect[20:] when doing batches
    plt.title("Accuracy of Gender Models in Test Folder")
    plt.savefig("/u/jyc9fyf/hamPlots/sexTestPlot")


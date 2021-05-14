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
    wanted = (df.sex == value)
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
    
def get_stats(mainmodel, dataloader, return_acts=True):

    all_acts = [] if return_acts else None
    activationCount = 0

    for (x, y, (age,sex)) in (dataloader):

        y_ = y.cuda()

        #print(x.shape)


        #acts = mainmodel(x.cuda()).detach()[:, 0]
        acts = mainmodel(x.cuda(), latent=0).detach()

        activationCount = ch.sum(acts > 0, 1).cpu().numpy()

        print(acts.shape)
        print(activationCount.shape)
        #incorrect = ((preds >= 0) != y_)
        #incorrectCount = incorrect.sum().item()
        #stats.append(y[incorrect].cpu().numpy())
        
        #print(acts)
        if return_acts:
            #all_acts.append(acts.cpu().numpy())
            #all_stats.append(y.cpu().numpy())
            all_acts.append(activationCount)

        

    all_acts = np.concatenate(all_acts)

    print(all_acts)

    return all_acts


def run_tests():
    allIncorrect = []
    #Used to get each set of activations from each model
    activations1 = []
    activations2 = []
    activations3 = []
    activations4 = []

    averageIncorrect = 0
    totalIncorrect = 0

    for FOLDER in tqdm(folder_paths):
        model_preds = []
        #averageIncorrect = totalIncorrect / 10
        #print(averageIncorrect)
        totalIncorrect = 0 
        
            

        print(FOLDER)
        #wanted_model = os.listdir(FOLDER)[:10]
        #wanted_model = os.listdir(FOLDER)[:10]

        for path in os.listdir(FOLDER)[:1]:
        

            MODELPATH = os.path.join(FOLDER, path)

            model = data_utils.BoneModel(1024)
            model = model.cuda()

            model.load_state_dict(ch.load(MODELPATH))
            model.eval()

            features = ch.load(os.path.join(base, "split_2/features_val.pt"))


            df1 = data_utils.BoneDataset(df, features, processed=True)

            test_loader = data_utils.DataLoader(
                df1, batch_size=batch_size * 2,
                shuffle=False, num_workers=2)

           
            activations = get_stats(model, test_loader, return_acts = True)

            #allIncorrect.append(incorrect)
            #print(incorrect)
            #totalIncorrect += incorrect
            #model_preds.append(preds)
            model_name.append(FOLDER)
            #if path == os.listdir(FOLDER)[9]: #If last model in directory
                #averageIncorrect = totalIncorrect / 10
                #print(totalIncorrect)
                #print(averageIncorrect)

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
    base = "/u/jyc9fyf/fnb/boneage/data/"
    incorrectSex1 = []
    incorrectSex0 = []
    incorrectAge1 = []
    incorrectAge0 = []
    
    activate1 = []
    activate2 = []
    activate3 = []
    activate4 = []

    df = process_data_sex(base,value = 1) #Assuming df_victim is unnecessary       #############
    batch_size = 250 #Any number works

    folder_paths = [
        "/p/adversarialml/as9rw/models_boneage/split_2/0.2", #test models in folders
        "/p/adversarialml/as9rw/models_boneage/split_2/0.3",
        "/p/adversarialml/as9rw/models_boneage/split_2/0.4",
        "/p/adversarialml/as9rw/models_boneage/split_2/0.5",
        "/p/adversarialml/as9rw/models_boneage/split_2/0.6",
        "/p/adversarialml/as9rw/models_boneage/split_2/0.7",
        "/p/adversarialml/as9rw/models_boneage/split_2/0.8"
    ]
    model_name = []

    activate1,activate2,activate3,activate4 = run_tests()

    #df = process_data_sex(base,value = 0)
    #activate1,activate2,activate3,activate4 = run_tests()

    #df = process_data_age(base, value = 1)
    #activate1,activate2,activate3,activate4 = run_tests()

    #df = process_data_age(base, value = 0)
    #activate1,activate2,activate3,activate4 = run_tests()



    #Graphing
    #x = ["Age0.2", "Age0.8", "Sex0.2", "Sex0.8"]
    #print(incorrectSex1)
    #print(incorrectSex0)
    #print(incorrectAge1)
    #print(incorrectAge0)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(np.sort(accuracies), np.arange(100))

    plt.title("Activations of Models in Test Folder on Sex = 1") 
    #plt.legend(loc='upper right')

    plt.savefig("/u/jyc9fyf/hamPlots/ActivationsPlot4Sex1")


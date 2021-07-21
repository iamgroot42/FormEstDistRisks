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
    

# tests the models and returns predictions and the total number of errors for a model
def get_stats(mainmodel, dataloader, return_preds=True):

    all_preds = [] if return_preds else None
    incorrectCount = 0
    for (x, y, (age,sex)) in (dataloader):

        y_ = y.cuda()


        preds = mainmodel(x.cuda()).detach()[:, 0]
        incorrect = ((preds >= 0) != y_)
        incorrectCount = incorrect.sum().item()
        if return_preds:
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    

    return all_preds, incorrectCount


#runs tests
def run_tests():
    allIncorrect = []
    graphIncorrect = []
    averageIncorrect = 0
    totalIncorrect = 0

    #loops through and tests models in the paths using get_stats
    for FOLDER in tqdm(folder_paths):
        model_preds = []
        #averageIncorrect = totalIncorrect / 10
        #print(averageIncorrect)
        totalIncorrect = 0 
        
            

        print(FOLDER)
        #wanted_model = os.listdir(FOLDER)[:10]
        #wanted_model = os.listdir(FOLDER)[:10]

        for path in os.listdir(FOLDER)[:100]:
        

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
            if path == os.listdir(FOLDER)[99]: #If last model in directory (change the value if using more models)
                averageIncorrect = totalIncorrect / 100
                print(totalIncorrect)
                print(averageIncorrect)
                graphIncorrect.append(averageIncorrect)



    return graphIncorrect
            

if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/" #dataset path
    incorrectSex1 = []
    incorrectSex0 = []
    incorrectAge1 = []
    incorrectAge0 = []


    df = process_data_sex(base,value = 1)
    batch_size = 250 * 8 * 4 #Any number works

    #model paths
    folder_paths = [
        "/u/jyc9fyf/hamModels/hamAge/testAge0.2",
        "/u/jyc9fyf/hamModels/hamAge/testAge0.8",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.2",
        "/u/jyc9fyf/hamModels/hamSex/testSex0.8"
    ]
    model_name = []

    incorrectSex1 = run_tests()

    df = process_data_sex(base,value = 0)
    incorrectSex0 = run_tests()

    df = process_data_age(base, value = 1)
    incorrectAge1 = run_tests()

    df = process_data_age(base, value = 0)
    incorrectAge0 = run_tests()

                    


    #Graphing
    x = ["Age0.2", "Age0.8", "Sex0.2", "Sex0.8"]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    #Make a scatterplot for test data for 100 models
    ax1.scatter(x, incorrectSex1, c = 'blue') 
    ax1.scatter(x, incorrectSex0, c = 'red')
    ax1.scatter(x, incorrectAge1, c = 'green')
    ax1.scatter(x, incorrectAge0, c = 'black')

    plt.title("Accuracy of Models in Test Folder") 


    plt.savefig("/u/jyc9fyf/hamPlots/TestPlot")


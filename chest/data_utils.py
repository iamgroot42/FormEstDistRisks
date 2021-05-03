from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch as ch
import pandas as pd
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
import utils


class ChestDataset(Dataset):
    def __init__(self, df, argument=None, processed=False):
        if processed:
            self.features = argument
        else:
            self.transform = argument
        self.processed = processed
        self.df = df

        print(np.mean(df['label']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.processed:
            X = self.features[self.df['path'].iloc[index]]
        else:
            # X = Image.open(self.df['path'][index])
            X = Image.open(self.df['path'][index]).convert('RGB')
            if self.transform:
                X = self.transform(X)

        y = ch.tensor(int(self.df['label'][index]))
        gender = ch.tensor(int(self.df['gender'][index]))
        age = ch.tensor(int(self.df['age'][index]))

        return X, y, (gender, age)


class ChestWrapper:
    def __init__(self, df_train, df_val, features=None):
        self.df_train = df_train
        self.df_val = df_val
        self.input_size = 299
        data_transform = transforms.Compose([
            transforms.Resize(350),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

        if features is None:
            self.ds_train = ChestDataset(self.df_train, data_transform)
            self.ds_val = ChestDataset(self.df_val, data_transform)
        else:
            self.ds_train = ChestDataset(
                self.df_train, features["train"], processed=True)
            self.ds_val = ChestDataset(
                self.df_val, features["val"], processed=True)

    def get_loaders(self, batch_size, shuffle=False):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        val_loader = DataLoader(
            self.ds_val, batch_size=batch_size * 2,
            shuffle=shuffle, num_workers=2)

        return train_loader, val_loader


def stratified_df_split(df, second_ratio):
    # Get new column for stratification purposes
    def fn(row): return str(row.gender) + str(row.age) + str(row.label)
    col = df.apply(fn, axis=1)
    df = df.assign(stratify=col.values)

    stratify = df['stratify']
    df_1, df_2 = train_test_split(
        df, test_size=second_ratio,
        stratify=stratify)

    # Delete remporary stratification column
    df.drop(columns=['stratify'], inplace=True)
    df_1 = df_1.drop(columns=['stratify'])
    df_2 = df_2.drop(columns=['stratify'])

    return df_1.reset_index(), df_2.reset_index()


def stratified_patient_df_split(df, second_ratio, iters=2000):
    patients_all = df["Patient ID"].unique()

    def get_stratification_vector(x):
        return np.array([np.mean(x.gender), np.mean(x.age), np.mean(x.label)])

    picked_1, picked_2, best = None, None, np.inf

    # Keep generating patient-ID divisions until similar ratios are achieved
    iterator = tqdm(range(iters))
    for i in iterator:
        rds = np.random.permutation(len(patients_all))
        split_point = int(second_ratio * len(rds))
        patients_1 = patients_all[rds[split_point:]]
        patients_2 = patients_all[rds[:split_point]]

        df_1 = df.loc[df['Patient ID'].isin(patients_1)]
        df_2 = df.loc[df['Patient ID'].isin(patients_2)]

        v_1 = get_stratification_vector(df_1)
        v_2 = get_stratification_vector(df_2)
        dist = np.linalg.norm(v_1 - v_2)

        if dist < best:
            best = dist
            picked_1 = df_1
            picked_2 = df_2

        iterator.set_description("Best stratified loss: %.5f" % best)

    # Delete remporary stratification column
    df_1 = picked_1.reset_index()
    df_2 = picked_2.reset_index()

    return df_1, df_2


def process_data(path, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, 'Data_Entry_2017.csv'))

    # Get data together
    all_image_paths = {os.path.basename(x): x for x in
                       glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}

    # Check how many files found
    print('Scans found: %d/%d images' % (len(all_image_paths), df.shape[0]))
    df['path'] = df['Image Index'].map(all_image_paths.get)

    # Convert gender to 0/1
    df['gender'] = df['Patient Gender'].map(lambda x: 1*(x == 'F'))

    # Convert age to 0/1
    df['age'] = df['Patient Age'].map(lambda x: 1 * (x > 48))

    # Treat finding/no-finding as label for task
    df['Finding Labels'] = df['Finding Labels'].map(
        lambda x: x.replace('No Finding', ''))
    df['label'] = df['Finding Labels'].map(lambda x: 1 * (x != ''))

    # Return stratified split
    return stratified_patient_df_split(df, split_second_ratio)


def useful_stats(df):
    print("%d | %.2f | %.2f | %.2f" % (
        len(df),
        df["label"].mean(),
        df["age"].mean(),
        df["gender"].mean()))


if __name__ == "__main__":
    DATA_DIR = "./data"

    # Make a 2:1 victim:adv split
    df_victim, df_adv = process_data(DATA_DIR, split_second_ratio=1/3)

    df_victim_train, df_victim_test = stratified_patient_df_split(df_victim, 0.2)
    df_adv_train, df_adv_test = stratified_patient_df_split(df_adv, 0.2)

    # Save these splits
    def save_split(df, split):
        useful_stats(df)
        print()

        # Get train-val splits
        train_df, val_df = stratified_df_split(df, 0.2)

        # Ensure directory exists
        dir_prefix = "./data/splits/%d/" % split
        utils.ensure_dir_exists(dir_prefix)

        # Save train-test splits
        train_df.to_csv(os.path.join(dir_prefix, "train.csv"))
        val_df.to_csv(os.path.join(dir_prefix, "val.csv"))

    save_split(df_victim, 1)
    save_split(df_adv, 2)

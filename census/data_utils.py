import utils
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import requests
import pandas as pd
import os
from tqdm import tqdm


BASE_DATA_DIR = "<PATH_TO_DATASET>"

SUPPORTED_PROPERTIES = ["sex", "race", "none"]
PROPERTY_FOCUS = {"sex": "Female", "race": "White"}
SUPPORTED_RATIOS = ["0.0", "0.1", "0.2", "0.3",
                    "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]


# US Income dataset
class CensusIncome:
    def __init__(self, path=BASE_DATA_DIR):
        self.urls = [
            "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        ]
        self.columns = [
            "age", "workClass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        self.dropped_cols = ["education", "native-country"]
        self.path = path
        self.download_dataset()
        # self.load_data(test_ratio=0.4)
        self.load_data(test_ratio=0.5)

    # Download dataset, if not present
    def download_dataset(self):
        if not os.path.exists(self.path):
            print("==> Downloading US Census Income dataset")
            os.mkdir(self.path)
            print("==> Please modify test file to remove stray dot characters")

            for url in self.urls:
                data = requests.get(url).content
                filename = os.path.join(self.path, os.path.basename(url))
                with open(filename, "wb") as file:
                    file.write(data)

    # Process, handle one-hot conversion of data etc
    def process_df(self, df):
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = ['workClass', 'occupation', 'race', 'sex',
                    'marital-status', 'relationship']
        # Drop columns that do not help with task
        df = df.drop(columns=self.dropped_cols, axis=1)
        # Club categories not directly relevant for property inference
        df["race"] = df["race"].replace(
            ['Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other')
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        # Drop features pruned via feature engineering
        prune_feature = [
            "workClass:Never-worked",
            "workClass:Without-pay",
            "occupation:Priv-house-serv",
            "occupation:Armed-Forces"
        ]
        df = df.drop(columns=prune_feature, axis=1)
        return df

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        Y = P['income'].to_numpy()
        X = P.drop(columns='income', axis=1)
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop, custom_limit=None):

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to data
            TRAIN_DF = get_filter(TRAIN_DF, filter_prop,
                                  split, prop_ratio, is_test=0,
                                  custom_limit=custom_limit)
            TEST_DF = get_filter(TEST_DF, filter_prop,
                                 split, prop_ratio, is_test=1,
                                 custom_limit=custom_limit)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(
                TRAIN_DF), self.get_x_y(TEST_DF)

            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    # Create adv/victim splits, normalize data, etc
    def load_data(self, test_ratio, random_state=42):
        # Load train, test data
        train_data = pd.read_csv(os.path.join(self.path, 'adult.data'),
                                 names=self.columns, sep=' *, *',
                                 na_values='?', engine='python')
        test_data = pd.read_csv(os.path.join(self.path, 'adult.test'),
                                names=self.columns, sep=' *, *', skiprows=1,
                                na_values='?', engine='python')

        # Add field to identify train/test, process together
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        df = pd.concat([train_data, test_data], axis=0)
        df = self.process_df(df)

        # Take note of columns to scale with Z-score
        z_scale_cols = ["fnlwgt", "capital-gain", "capital-loss"]
        for c in z_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        # Take note of columns to scale with min-max normalization
        minmax_scale_cols = ["age",  "hours-per-week", "education-num"]
        for c in minmax_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].min()) / df[c].max()

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']
                                         == 1], df[df['is_train'] == 0]

        # Drop 'train/test' columns
        self.train_df = self.train_df.drop(columns=['is_train'], axis=1)
        self.test_df = self.test_df.drop(columns=['is_train'], axis=1)

        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=test_ratio,
                                         random_state=rs)
            # Stratification on the properties we care about for this dataset
            # so that adv/victim split does not introduce
            # unintended distributional shift
            splitter = sss.split(
                this_df, this_df[["sex:Female", "race:White", "income"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        # Create train/test splits for victim/adv
        self.train_df_victim, self.train_df_adv = s_split(self.train_df)
        self.test_df_victim, self.test_df_adv = s_split(self.test_df)


def get_filter(df, filter_prop, split, ratio, is_test, custom_limit=None):
    if filter_prop == "none":
        return df
    elif filter_prop == "sex":
        def lambda_fn(x): return x['sex:Female'] == 1
    elif filter_prop == "race":
        def lambda_fn(x): return x['race:White'] == 1
    prop_wise_subsample_sizes = {
        "adv": {
            "sex": (1100, 500),
            "race": (2000, 1000),
        },
        "victim": {
            "sex": (1100, 500),
            "race": (2000, 1000),
        },
    }

    if custom_limit is None:
        subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
    else:
        subsample_size = custom_limit
    return utils.heuristic(df, lambda_fn, ratio,
                           subsample_size, class_imbalance=3,
                           n_tries=100, class_col='income',
                           verbose=False)


def cal_q(df, condition):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    return len(qualify), len(notqualify)


def get_df(df, condition, x):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    np.random.shuffle(qualify)
    return df.iloc[qualify[:x]]


def cal_n(df, con, ratio):
    q, n = cal_q(df, con)
    current_ratio = q / (q+n)
    # If current ratio less than desired ratio, subsample from non-ratio
    if current_ratio <= ratio:
        if ratio < 1:
            nqi = (1-ratio) * q/ratio
            return q, nqi
        return q, 0
    else:
        if ratio > 0:
            qi = ratio * n/(1 - ratio)
            return qi, n
        return 0, n

# Wrapper for easier access to dataset
class CensusWrapper:
    def __init__(self, filter_prop="none", ratio=0.5, split="all"):
        self.ds = CensusIncome()
        self.split = split
        self.ratio = ratio
        self.filter_prop = filter_prop

    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.filter_prop,
                                custom_limit=custom_limit)

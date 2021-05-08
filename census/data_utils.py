import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from robustness.tools.misc import log_statement
from sklearn import preprocessing
import requests
import pandas as pd
import os


BASE_DATA_DIR = "/u/as9rw/work/fnb/census/census_data"
SUPPORTED_PROPERTIES = ["sex", "race", "age"]


# US Income dataset
class CensusIncome:
    def __init__(self, path=BASE_DATA_DIR):
        self.urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
        self.columns = ["age", "workClass", "fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship",
                        "race", "sex", "capital-gain", "capital-loss",
                        "hours-per-week", "native-country", "income"]
        self.path = path
        self.download_dataset()

    def download_dataset(self):
        if not os.path.exists(self.path):
            log_statement("==> Downloading US Census Income dataset")
            os.mkdir(self.path)
            log_statement(
                "==> Please modify test file to remove stray dot characters")

            for url in self.urls:
                data = requests.get(url).content
                filename = os.path.join(self.path, os.path.basename(url))
                with open(filename, "wb") as file:
                    file.write(data)

    def process_df(self, df):
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        def oneHotCatVars(x, colname):
            df_1 = df.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(df[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = ['workClass', 'occupation', 'race', 'sex',
                    'marital-status', 'relationship', 'native-country']
        # Drop education
        df = df.drop(columns='education', axis=1)
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        return df

    def load_data(self,
                  train_filter=None,
                  test_ratio=0.75,
                  random_state=42,
                  split="all",
                  sample_ratio=1.):
        train_data = pd.read_csv(os.path.join(self.path, 'adult.data'),
                                 names=self.columns, sep=' *, *',
                                 na_values='?', engine='python')
        test_data = pd.read_csv(os.path.join(self.path, 'adult.test'),
                                names=self.columns, sep=' *, *', skiprows=1,
                                na_values='?', engine='python')

        # Add field to identify train/test, process together, split back
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        df = pd.concat([train_data, test_data], axis=0)
        df = self.process_df(df)

        train_df, test_df = df[df['is_train'] == 1], df[df['is_train'] == 0]

        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=test_ratio,
                                         random_state=rs)
            splitter = sss.split(
                this_df, this_df[["sex:Female", "race:White", "income"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        train_df_first, train_df_second = s_split(train_df)
        test_df_first, test_df_second = s_split(test_df)

        # Further sample data, if requested
        if sample_ratio < 1:
            # Don't fixed random seed (random sampling wanted)
            # In this mode, train and test data are combined
            train_df_first = pd.concat([train_df_first, test_df_first])
            train_df_second = pd.concat([train_df_second, test_df_second])
            _, train_df_first = s_split(train_df_first, None)
            _, train_df_second = s_split(train_df_second, None)

        def get_x_y(P):
            # Scale X values
            Y = P['income'].to_numpy()
            X = P.drop(columns='income', axis=1)
            cols = X.columns
            X = X.to_numpy()
            return (X.astype(float), np.expand_dims(Y, 1), cols)

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to train data
            # Efectively using different distribution to train it
            if train_filter is not None:
                TRAIN_DF = train_filter(TRAIN_DF)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = get_x_y(
                TRAIN_DF), get_x_y(TEST_DF)
            # Preprocess data (scale)
            X = np.concatenate((x_tr, x_te), 0)
            X = preprocessing.scale(X)
            x_tr = X[:x_tr.shape[0]]
            x_te = X[x_tr.shape[0]:]

            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(train_df, test_df)
        if split == "victim":
            return prepare_one_set(train_df_first, test_df_first)
        return prepare_one_set(train_df_second, test_df_second)

    def split_on_col(self, x, y, cols,
                     second_ratio, seed=42, debug=False):
        # Temporarily combine cols and y for stratification
        y_ = np.concatenate((y, x[:, cols]), 1)
        x_1, x_2, y_1, y_2 = train_test_split(x,
                                              y_,
                                              test_size=second_ratio,
                                              random_state=seed)
        # If debugging enabled, print ratios of col_index
        # Before and after splits
        if debug:
            print("Before:", np.mean(x[:, cols], 0), np.mean(y))
            print("After:", np.mean(x_1[:, cols], 0), np.mean(y_1[:, 0]))
            print("After:", np.mean(x_2[:, cols], 0), np.mean(y_2[:, 0]))
        # Get rid of extra column in y
        y_1, y_2 = y_1[:, 0], y_2[:, 0]
        # Return split data
        return (x_1, y_1), (x_2, y_2)


def sex_filter(ratio=0.65, verbose=False):
    def func(df):
        return utils.filter(
            df, lambda x: x['sex:Female'] == 1, ratio, verbose)
    return func


def race_filter(ratio=1.0, verbose=False):
    def func(df):
        return utils.filter(
            df, lambda x: x['race:White'] == 0, ratio, verbose)
    return func


def income_filter(ratio=0.5, verbose=False):
    def func(df):
        return utils.filter(
            df, lambda x: x['income'] == 1, ratio, verbose)
    return func


def get_default_filter(prop, ratio, verbose=False):
    if prop == "sex":
        return sex_filter(ratio, verbose)
    elif prop == "race":
        return race_filter(ratio, verbose)
    elif prop == "income":
        return income_filter(ratio, verbose)
    elif prop == "none":
        return None
    else:
        raise ValueError("Invalid filter requested")


class CensusWrapper:
    def __init__(self, filter=None, split="all"):
        if filter not in SUPPORTED_PROPERTIES:
            raise ValueError("Invalid property specified")

        if filter == "sex":
            self.filter = sex_filter
        elif filter == "race":
            self.filter = race_filter
        elif filter == "income":
            self.filter = income_filter
        else:
            # Not one of the default properties, use given filter
            self.filter = filter

        self.ds = CensusIncome()
        self.split = split

    def load_data(self):
        return self.ds.load_data(train_filter=self.filter,
                                 split=self.split)

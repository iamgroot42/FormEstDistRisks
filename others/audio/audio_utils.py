from HTK import HTKFile
from tqdm import tqdm
import os
import numpy as np


class VoxForgeData:
    def __init__(self):
        self.age = {
            0: [
                'ADULT',
                'ADULT (BORN IN 1983)',
                'ADULTO',
                '[ADULT]'
                ],
            1: [
                'YOUTH',
                '[YOUTH]'
            ],
            2: [
                'SENIOR'
            ]
        }
        self.sex = {
            0: [
                'FEMALE',
                '[FEMALE]'
            ],
            1: [
                'MALE',
                'MAKE',
                '[MALE]',
            ]
        }
        self.property = ['INDIAN ENGLISH', '[INDIAN ENGLISH]']
        self.property_reject = ['PLEASE SELECT', 'UNKNOWN']

    def load_data(self, path):
        self.X, self.Y, Z = np.load(path, allow_pickle=True)
        age, sex, dialect = [], [], []
        indices = []
        for i, z in enumerate(Z):
            matched = False
            for k, v in self.age.items():
                if z[0].upper() in v:
                    age_ = k
                    matched = True
                    break

            # Ignore if outlier
            if not matched:
                continue

            matched = False
            for k, v in self.sex.items():
                if z[1].upper() in v:
                    sex_ = k
                    matched = True
                    break

            # Ignore if outlier
            if not matched:
                continue

            dial = z[2].upper().rstrip(".")
            # Do not consider datum if dialect not known
            if dial in self.property_reject:
                continue

            age.append(age_)
            sex.append(sex_)
            dialect.append(1 * (dial in self.property))
            indices.append(i)

        self.X = self.X[indices]
        self.Y = self.Y[indices]
        self.Z = np.array([age, sex, dialect]).T

    def flatten_data_all(self):
        X, Y, Z = [], [], []

        for (x, y, z) in zip(self.X, self.Y, self.Z):
            for x_ in x:
                X.append(x_)
            for y_ in y:
                Y.append(y_)
            for _ in range(len(y)):
                Z.append(z)

        # Sanity check
        assert len(X) == len(Y) and len(Y) == len(Z)
        return X, Y, Z


class VoxForgeReader:
    def __init__(self, root):
        self.root = root
        self.wanted_info = [
            "AGE RANGE:",
            "GENDER:",
            "SEX:",
            "PRONUNCIATION DIALECT:",
            "PRONUNICATION DIALECT:",
            "MICROPHONE TYPE:",
        ]

    def extract_readme_info(self, path, log_errors):
        if not os.path.exists(path):
            return None

        with open(path, 'r') as f:
            lines = f.read().splitlines()

        attrs = []
        try:
            for i, attr in enumerate(self.wanted_info):
                wi = filter(lambda x: x.strip("\t ").upper().startswith(attr),
                            lines)
                wi_list = list(wi)
                if len(wi_list) > 0:
                    info = wi_list[0].split(":", 1)[1].strip()
                    attrs.append(info.rstrip(';'))

            # if log_errors and len(attrs) != 3:
            if log_errors and len(attrs) != 4:
                # If crucial attributes are not available for file
                # Do not use these recordings
                raise ValueError("Not all attributes were found!")
                return None

        except Exception as e:
            print(e)
            print(wi_list)
            exit(0)
            return None

        return attrs

    def parse_mfcc_file(self, path):
        try:
            htk_reader = HTKFile()
            htk_reader.load(path, warning=False)
            return np.array(htk_reader.data)
        except:
            print("Problem with %s" % path)
            return None

    def get_prompts(self, path):
        if not os.path.exists(path):
            return None

        with open(path, 'r') as f:
            lines = f.read().splitlines()

        prompt_dict = {}
        for x in lines:
            k, v = x.split(" ", 1)
            prompt_dict[k] = v

        return prompt_dict

    def get_speaker_data(self, path, log_errors=False):
        if not os.path.exists(os.path.join(path, "etc")):
            return None
        fileList = os.listdir(os.path.join(path, "etc"))

        def check_exists(L):
            for l in L:
                if l in fileList:
                    return l
            return None

        # Make sure README file exists
        readmeName = check_exists(["README", "readme"])
        if readmeName is None:
            return None

        # Make sure PROMPTS file exists
        promptName = check_exists(["PROMPTS", "prompts"])
        if promptName is None:
            return None

        # Get speaker information
        rpath = os.path.join(path, os.path.join("etc", readmeName))
        aux = self.extract_readme_info(rpath, log_errors)
        if aux is None:
            print(path)
            print("Yikes Aux")
            return None

        # Get prompts
        ppath = os.path.join(path, os.path.join("etc", promptName))
        Y = self.get_prompts(ppath)
        if Y is None:
            print(path)
            print("Yikes Y")
            return None

        # Get MFCC features for recordings
        X = {}
        mfc_dir = os.path.join(path, "mfc")
        mfc_files = filter(lambda x: x.endswith(".mfc"),
                           os.listdir(mfc_dir))
        for mfc_file in mfc_files:
            mfcc = self.parse_mfcc_file(os.path.join(mfc_dir, mfc_file))
            X[mfc_file.rsplit(".", 1)[0]] = mfcc

        # Align audio files and prompts
        X_act, Y_act = [], []
        for y, yat in Y.items():
            x_eff = y.split("/")[-1]
            xat = X[x_eff]
            if xat is not None:
                X_act.append(xat)
                Y_act.append(yat)

        X_act = np.array(X_act)
        Y_act = np.array(Y_act)

        return X_act, Y_act, aux

    def load_data_raw(self,
                      log_errors=False):
        data = {}
        speakers = os.listdir(self.root)
        for sid in tqdm(speakers):
            identity_dir = os.path.join(self.root, sid)
            datum = self.get_speaker_data(identity_dir,
                                          log_errors)
            if datum is not None:
                data[sid] = datum

        return data

    def load_data_np(self, dump_it_all=None):
        data = []
        speakers = os.listdir(self.root)
        speakers = list(map(lambda x: os.path.join(self.root, x), speakers))

        for identity_file in tqdm(speakers):
            datum = np.load(identity_file, allow_pickle=True)
            data.append(datum)

        X, Y, Z = [], [], []
        for d in data:
            X.append(d[0])
            Y.append(d[1])
            Z.append(np.array(d[2]))

        if dump_it_all is not None:
            print("Saving all data into one file")
            np.save(dump_it_all, np.array([X, Y, Z]))

        return data

    def dump_numpy(self, dest_dir):
        data = self.load_data_raw()

        for sid, datum in data.items():
            np_datum = np.array(datum)
            np.save(os.path.join(dest_dir, sid),
                    np_datum)

    def load_data(self, numpy=True, log_errors=False):
        if numpy:
            return self.load_data_np()
        return self.load_data_raw()


if __name__ == "__main__":
    # ds = VoxForgeReader("./data_np")
    # ds.load_data_np("./data_np_single")
    d = VoxForgeData()
    d.load_data("./data_np_single.npy")
    x, y, z = d.flatten_data_all()
    print(x[0].shape)
    print(y[0])
    print(z[0])

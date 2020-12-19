from HTK import HTKFile
from tqdm import tqdm

import os
import numpy as np


class VoxForge:
    def __init__(self, root):
        self.root = root
        self.wanted_info = [
            "AGE RANGE:",
            "GENDER:",
            "SEX:",
            "PRONUNCIATION DIALECT:",
            "PRONUNICATION DIALECT:"
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

            if log_errors and len(attrs) != 3:
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

    def load_data_np(self):
        data = []
        speakers = os.listdir(self.root)
        for sid in tqdm(speakers):
            identity_file = os.path.join(self.root, sid)
            datum = np.load(identity_file, allow_pickle=True)
            data.append(datum)

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
    # ds = VoxForge("./data")
    # ds.dump_numpy("./data_np")
    ds_ = VoxForge("./data_np")
    data = ds_.load_data_np()
    print(len(data))

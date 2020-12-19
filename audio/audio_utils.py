from HTK import HTKFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np


class VoxForge:
    def __init__(self, root):
        self.root = root
        self.wanted_info = [
            "AGE RANGE:",
            "GENDER:",
            "SEX:",
            # "LANGUAGE:",
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
                elif i < 2:
                    # If crucial attributes are not available for file
                    # Do not use these recordings
                    if log_errors:
                        print("%s not found in file %s" % (attr, path))
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
        # Get MFCC features for recordings
        X = {}
        mfc_dir = os.path.join(path, "mfc")
        mfc_files = filter(lambda x: x.endswith(".mfc"),
                           os.listdir(mfc_dir))
        for mfc_file in mfc_files:
            mfcc = self.parse_mfcc_file(os.path.join(mfc_dir, mfc_file))
            X[mfc_file] = mfcc

        # Get speaker information
        rpath = os.path.join(path, os.path.join("etc", "README"))
        aux = self.extract_readme_info(rpath, log_errors)

        # If README is not a file, try readme
        if aux is None:
            rpath = os.path.join(path, os.path.join("etc", "readme"))
            aux = self.extract_readme_info(rpath, log_errors)
            # If even readme is not a file, flag it
            if aux is None:
                return None

        # Get prompts
        ppath = os.path.join(path, os.path.join("etc", "PROMPTS"))
        Y = self.get_prompts(ppath)

        # If PROMPTS is not a file, try prompts
        if Y is None:
            ppath = os.path.join(path, os.path.join("etc", "prompts"))
            Y = self.get_prompts(ppath)
            # If even prompts is not a file, flag it
            if Y is None:
                return None

        return X, Y, aux

    def load_data(self,
                  n_threads=8,
                  log_errors=False):
        data = []
        from tqdm import tqdm
        with ThreadPoolExecutor(n_threads) as executor:
            futures = []

            for identity in os.listdir(self.root):
                identity_dir = os.path.join(self.root, identity)
                futures.append(executor.submit(
                    self.get_speaker_data,
                    identity_dir,
                    log_errors,))

            for x in tqdm(as_completed(futures)):
                datum = x.result()
                if datum is not None:
                    data.append(datum)

        return data


if __name__ == "__main__":
    ds = VoxForge("data")
    data = ds.load_data(n_threads=32)
    print(len(data))

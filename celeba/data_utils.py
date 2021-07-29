from torchvision import transforms
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import utils
from tqdm import tqdm
from PIL import Image
import pandas as pd
from utils import worker_init_fn


BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/celeba"
# BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/"
PRESERVE_PROPERTIES = ['Smiling', 'Young', 'Male', 'Attractive']
SUPPORTED_PROPERTIES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]


def get_bboxes():
    fpath = os.path.join(BASE_DATA_DIR, "list_bbox_celeba.txt")
    boxes = pd.read_csv(fpath, delim_whitespace=True, header=1, index_col=0)
    return boxes


def get_identities():
    fpath = os.path.join(BASE_DATA_DIR, "identity_CelebA.txt")
    identity = pd.read_csv(fpath, delim_whitespace=True,
                           header=None, index_col=0)
    return np.array(identity.values).squeeze(1)


def get_splits():
    fpath = os.path.join(BASE_DATA_DIR, "list_eval_partition.txt")
    splits = pd.read_csv(fpath, delim_whitespace=True,
                         header=None, index_col=0)
    return splits


def victim_adv_identity_split(identities, attrs, n_tries=1000, adv_ratio=0.25):
    # Create mapping of IDs to face images
    mapping = {}
    for i, id_ in enumerate(identities):
        mapping[id_] = mapping.get(id_, []) + [i]

    # Group-check for attribute values
    def get_vec(spl):
        picked_keys = np.array(list(mapping.keys()))[spl]
        collected_ids = np.concatenate([mapping[x] for x in picked_keys])
        vals = [attrs[pp].iloc[collected_ids].mean()
                for pp in PRESERVE_PROPERTIES]
        return np.array(vals)

    # Take note of original ratios
    ratios = np.array([attrs[pp].mean() for pp in PRESERVE_PROPERTIES])

    iterator = tqdm(range(n_tries))
    best_splits = None, None
    best_diff = (np.inf, np.inf)
    for _ in iterator:
        # Generate random victim/adv split
        randperm = np.random.permutation(len(mapping))
        split_point = int(len(randperm) * adv_ratio)
        adv_ids, victim_ids = randperm[:split_point], randperm[split_point:]

        # Get ratios for these splits
        vec_adv = get_vec(adv_ids)
        vec_victim = get_vec(victim_ids)

        # Measure ratios for images contained in these splits
        diff_adv = np.linalg.norm(vec_adv-ratios)
        diff_victim = np.linalg.norm(vec_victim-ratios)

        if best_diff[0] + best_diff[1] > diff_adv + diff_victim:
            best_diff = (diff_adv, diff_victim)
            best_splits = adv_ids, victim_ids

        iterator.set_description(
            "Best ratio differences: %.4f, %.4f" % (best_diff[0], best_diff[1]))

    # Extract indices corresponding to splits
    split_adv, split_victim = best_splits

    picked_keys_adv = np.array(list(mapping.keys()))[split_adv]
    adv_mask = np.concatenate([mapping[x] for x in picked_keys_adv])

    picked_keys_victim = np.array(list(mapping.keys()))[split_victim]
    victim_mask = np.concatenate([mapping[x] for x in picked_keys_victim])

    return adv_mask, victim_mask


def get_attributes():
    fpath = os.path.join(BASE_DATA_DIR, "list_attr_celeba.txt")
    attrs = pd.read_csv(fpath, delim_whitespace=True, header=1)
    attrs = (attrs + 1) // 2
    attr_names = list(attrs.columns)
    return attrs, attr_names


def create_df(attr_dict, filenames):
    # Create DF from filenames to use heuristic for ratio-preserving splits
    all = []
    for filename in filenames:
        y = list(attr_dict[filename].values())
        all.append(y + [filename])
    df = pd.DataFrame(data=all, columns=SUPPORTED_PROPERTIES + ['filename'])
    return df


def ratio_sample_data(filenames, attr_dict, label_name,
                      prop, ratio, cwise_sample):
    # Make DF
    df = create_df(attr_dict, filenames)

    # Make filter
    def condition(x): return x[prop] == 1

    parsed_df = utils.heuristic(
                    df, condition, ratio,
                    cwise_sample, class_imbalance=1.0,
                    n_tries=100, class_col=label_name,
                    verbose=True)
    # Extract filenames from parsed DF
    return parsed_df["filename"].tolist()


class CelebACustomBinary(Dataset):
    def __init__(self, classify, filelist_path, attr_dict,
                 prop, ratio, cwise_sample,
                 shuffle=False, transform=None):
        self.transform = transform
        self.classify = classify
        self.attr_dict = attr_dict

        self.classify_index = SUPPORTED_PROPERTIES.index(classify)

        # Get filenames
        with open(filelist_path) as f:
            self.filenames = f.read().splitlines()

        # Sort to get deterministic order
        self.filenames.sort()

        # Apply requested filter
        self.filenames = ratio_sample_data(
            self.filenames, self.attr_dict,
            classify, prop, ratio, cwise_sample)

        if shuffle:
            np.random.shuffle(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        x = Image.open(os.path.join(
            BASE_DATA_DIR, "img_align_celeba", filename))
        y = np.array(list(self.attr_dict[filename].values()))

        if self.transform:
            x = self.transform(x)

        return x, y[self.classify_index], y


class CelebaWrapper:
    def __init__(self, prop, ratio, split,
                 classify="Smiling", augment=False,
                 cwise_samples=None):

        # Make sure specified label is valid
        if classify not in PRESERVE_PROPERTIES:
            raise ValueError("Specified label not available for images")

        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        if augment:
            augment_transforms = [
                transforms.RandomAffine(degrees=20,
                                        translate=(0.2, 0.2),
                                        shear=0.2),
                transforms.RandomHorizontalFlip()
            ]
            train_transforms = augment_transforms + train_transforms
        train_transforms = transforms.Compose(train_transforms)

        # Read attributes file to get attribute names
        attrs, _ = get_attributes()
        # Create mapping between filename and attributes
        attr_dict = attrs.to_dict(orient='index')

        # Use relevant file split information
        filelist_train = os.path.join(
            BASE_DATA_DIR, "splits", "75_25", split, "train.txt")
        filelist_test = os.path.join(
            BASE_DATA_DIR, "splits", "75_25", split, "test.txt")

        # Define number of sub-samples
        prop_wise_subsample_sizes = {
            "Smiling": {
                "adv": {
                    "Male": (10000, 1000),
                    "Attractive": (10000, 1200),
                    "Young": (6000, 600)
                },
                "victim": {
                    "Male": (15000, 3000),
                    "Attractive": (30000, 4000),
                    "Young": (15000, 2000)
                }
            },
            "Male": {
                "adv": {
                    "Young": (3000, 350),
                },
                "victim": {
                    "Young": (8000, 1400),
                }
            }
        }

        cwise_sample = prop_wise_subsample_sizes[classify][split][prop]
        if cwise_samples is not None:
            cwise_sample = cwise_samples

        self.ds_train = CelebACustomBinary(
            classify, filelist_train, attr_dict,
            prop, ratio, cwise_sample[0],
            transform=train_transforms,
            )
        self.ds_val = CelebACustomBinary(
            classify, filelist_test, attr_dict,
            prop, ratio, cwise_sample[1],
            transform=test_transforms)

    def get_loaders(self, batch_size, shuffle=True, eval_shuffle=False):
        num_workers = 16
        pff = 20
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=pff)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        test_loader = DataLoader(
            self.ds_val, batch_size=batch_size * 2,
            shuffle=eval_shuffle, num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=pff)

        return train_loader, test_loader


if __name__ == "__main__":
    # Load metadata files
    splits = get_splits()
    ids = get_identities()
    attrs, _ = get_attributes()
    filenames = np.array(splits.index.tolist())

    # 0 train, 1 validation, 2 test
    train_mask = np.logical_or(splits[1].values == 0, splits[1].values == 1)
    test_mask = splits[1].values == 2

    # Splits on test data
    test_adv, test_victim = victim_adv_identity_split(
        ids[test_mask], attrs[test_mask],
        n_tries=5000, adv_ratio=0.25)
    mask_locs = np.nonzero(test_mask)[0]
    test_adv_filenames = filenames[mask_locs[test_adv]]
    test_victim_filenames = filenames[mask_locs[test_victim]]

    # Splits on train data
    train_adv, train_victim = victim_adv_identity_split(
        ids[train_mask], attrs[train_mask],
        n_tries=5000, adv_ratio=0.25)
    mask_locs = np.nonzero(train_mask)[0]
    train_adv_filenames = filenames[mask_locs[train_adv]]
    train_victim_filenames = filenames[mask_locs[train_victim]]

    # Save split files for later use
    def save(data, path):
        with open(os.path.join(BASE_DATA_DIR, path), 'w') as f:
            f.writelines("%s\n" % l for l in data)

    save(test_adv_filenames, os.path.join(
        "splits", "75_25", "adv", "test.txt"))
    save(test_victim_filenames, os.path.join(
        "splits", "75_25", "victim", "test.txt"))
    save(train_adv_filenames, os.path.join(
        "splits", "75_25", "adv", "train.txt"))
    save(train_victim_filenames, os.path.join(
        "splits", "75_25", "victim", "train.txt"))

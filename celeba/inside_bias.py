import utils
import implem_utils

import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--sample', type=int, default=2500,
                        help='number of query points per label-property to use for calculations')
    parser.add_argument('--layer', type=int, default=6,
                        help="which layer's activations to look at")
    args = parser.parse_args()
    utils.flash_utils(args)

    batch_size = args.bs
    method_type = -args.layer

    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    folder_paths = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/",
        ]
    ]

    blind_test_models = [
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/10_0.928498243559719.pth",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/10_0.9093969555035128.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/15_0.9073793914943687.pth",
        ],
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/10_0.9240681998413958.pth",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/10_0.8992862807295797.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/20_0.8947974217311234.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/15_0.9120626151012892.pth",
        ]
    ]

    # Use existing dataset instead
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    target_prop = attrs.index("Smiling")
    all_x, all_y = [], []

    for index, UPFOLDER in enumerate(folder_paths):
        model_latents = []
        model_stats = []

        print()
        for pf in UPFOLDER:
            for MODELPATHSUFFIX in tqdm(os.listdir(pf)):
                # if not args.all:
                if not("18_" in MODELPATHSUFFIX or "13_" in MODELPATHSUFFIX):
                    continue
                # MODELPATH    = os.path.join(UPFOLDER, FOLDER, wanted_model)

                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                cropped_dataloader = DataLoader(td,
                                                batch_size=batch_size,
                                                shuffle=False)

                # Get latent representations
                latent, all_stats = implem_utils.get_features_for_model(cropped_dataloader,
                                                                        MODELPATH,
                                                                        method_type=method_type,
                                                                        weight_init=None)

                # Use only specified number of samples
                prop_attr = attrs.index("Attractive")
                wanted_indices = implem_utils.balanced_cut(all_stats,
                                                           prop_attr,
                                                           target_prop,
                                                           args.sample)
                latent = latent[wanted_indices]
                all_stats = all_stats[wanted_indices]

                # Get lambda values for these latent features
                lmbds = implem_utils.lambdas(latent)

                # Normalize across data for inter-model comparison
                lmbds -= np.min(lmbds)
                lmbds /= np.max(lmbds)

                # Compute lambda ratio
                yes_prop = np.nonzero(all_stats[:, prop_attr] == 1)[0]
                no_prop = np.nonzero(all_stats[:, prop_attr] == 0)[0]

                prop_vals = (np.mean(lmbds[yes_prop]), np.mean(lmbds[no_prop]))
                lambda_ratio = np.min(prop_vals) / np.max(prop_vals)

                print(pf)
                print("Lambda values:", prop_vals)
                print("Lambda ratio:", lambda_ratio)

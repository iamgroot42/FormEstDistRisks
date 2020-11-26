import numpy as np
import utils
import torch as ch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def calibration(latents, use_ref=None):
    headstart = 0 if use_ref else 1
    perms = []
    # Take first latent as reference
    # Pass through examples
    for i in range(latents.shape[1]):
        p_perm = []
        # get order of first model as reference
        mapping = use_ref if use_ref else {x: kk for kk, x in enumerate(np.argsort(latents[0, i]))} 
        for j in range(headstart, latents.shape[0]):
            p_perm.append([mapping.get(j) for j in np.argsort(latents[j, i])])
        perms.append(p_perm)
    perms = np.array(perms)

    # For each model
    weights = []
    for i in range(perms.shape[1]):
        # For each dimension
        weighted = np.zeros((perms.shape[2], perms.shape[2]))
        for j in range(perms.shape[2]):
            u, c = np.unique(perms[:, i, j], return_counts=True)
            weighted[j][u] = c
        weights.append(weighted.T)
    
    weights = np.array(weights) / perms.shape[0]
    # Right-multiplty latent with weights to get aligned versions
    return mapping, weights


def get_latents(mainmodel, dataloader):
    all_stats = []
    all_latent = []
    # for (x, y) in tqdm(dataloader, total=len(dataloader)):
    for (x, y) in dataloader:

        # latent = mainmodel(x.cuda(), deep_latent=6).detach()
        # latent = latent.mean(1)
        # latent = mainmodel(x.cuda(), deep_latent=14).detach()
        # latent = latent.view(latent.shape[0], -1)

        # Use scores
        # latent = mainmodel(x.cuda()).detach()[:, 0]
        latent = nn.sigmoid(mainmodel(x.cuda()).detach()[:, 0])
        # latent = mainmodel(x.cuda(), only_latent=True).detach()
        all_latent.append(latent.cpu().numpy())
        all_stats.append(y.cpu().numpy())

    all_latent = np.concatenate(all_latent, 0)
    all_stats = np.concatenate(all_stats)

    return all_latent, all_stats


def get_features_for_model(dataloader, MODELPATH, weight_init, layers=[64, 16]):
    # Load model
    model = utils.FaceModel(512, train_feat=True,
                            weight_init=weight_init, hidden=layers).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(MODELPATH), strict=False)
    model.eval()

    # Get latent representations
    lat, sta = get_latents(model, dataloader)
    # lat = np.sort(lat, 1)
    # lat = np.array([np.std(lat, 1), np.mean(lat == 0, 1), np.mean(lat, 1), np.mean(lat ** 2, 1)]).T
    return (lat, sta)


if __name__ == "__main__":

    batch_size = 1024 #128 #512
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
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/10_0.928498243559719.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/10_0.9093969555035128.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/20_0.9151053864168618.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/20_0.9108606557377049.pth"
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/10_0.9240681998413958.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/10_0.8992862807295797.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/20_0.9259516256938938.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/20_0.9042426645519429.pth"
        ]
    ]

    if 1 == 2:
        cropmodel = MTCNN(device='cuda')

        # Get all cropped images
        x_cropped, y_cropped = [], []
        _, dataloader = ds.make_loaders(
            batch_size=batch_size, workers=8, shuffle_train=False, shuffle_val=False, only_val=True)
        for x, y in tqdm(dataloader, total=len(dataloader)):
            x_, indices = utils.get_cropped_faces(cropmodel, x)
            x_cropped.append(x_.cpu())
            y_cropped.append(y[indices])

        # Make dataloader out of this filtered data
        x_cropped = ch.cat(x_cropped, 0)
        y_cropped = ch.from_numpy(np.concatenate(y_cropped, 0))
        td = TensorDataset(x_cropped, y_cropped)

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

        for pf in UPFOLDER:
            for MODELPATHSUFFIX in tqdm(os.listdir(pf)):
                # if not("3_" in MODELPATHSUFFIX or "10_" in MODELPATHSUFFIX or "20_" in MODELPATHSUFFIX): continue
                # MODELPATH    = os.path.join(UPFOLDER, FOLDER, wanted_model)

                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)

                # Get latent representations
                latent, all_stats = get_features_for_model(
                    cropped_dataloader, MODELPATH, weight_init=None)
                model_stats.append(all_stats)
                model_latents.append(latent)

                all_y.append(np.ones((latent.shape[0])) * index)

        model_latents = np.array(model_latents)
        model_stats = np.array(model_stats)

        all_x.append(model_latents)

    all_x = np.concatenate(np.array(all_x), 0)
    idxs = [np.random.permutation(all_x.shape[1])[:1000] for i in range(10)]
    # Calibrate at this point
    # cali, weights = calibration(all_x)
    # Calibrate all_x (except baseline first)
    # for i in range(weights.shape[0]): all_x[i+1] = np.matmul(all_x[i+1], weights[i])
    clfs = []

    # If using each point independently
    # all_x = np.concatenate(all_x, 0)
    # all_y = np.concatenate(all_y, 0)

    # Train 10 classifiers on random samples
    for i in range(10):
        # haha don't go brrr
        # x_tr, x_te, y_tr, y_te = train_test_split(all_x, all_y, test_size=0.33)

        # haha go brr
        num_each = all_x.shape[0] // 2
        x_tr, x_te, y_tr, y_te = train_test_split(all_x[:, idxs[i]], [0] * num_each + [1] * num_each, test_size=0.25)

        clf = MLPClassifier(hidden_layer_sizes=(128, 32))
        clf.fit(x_tr, y_tr)
        print("%.2f train, %.2f test" % (clf.score(x_tr, y_tr), clf.score(x_te, y_te)))

        clfs.append(clf)

    # Test out on unseen models
    all_scores = []
    for pc in blind_test_models:
        ac = []
        for path in pc:
            cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)
            latent, _ = get_features_for_model(
                cropped_dataloader, path, weight_init=None)#"vggface2")

            # Calibrate latent
            # _, weights = calibration(np.expand_dims(latent, 0), use_ref=cali)
            # latent = np.matmul(latent, weights[0])

            # preds = [clf.predict_proba(latent[idx])[:, 1] for idx, clf in zip(idxs, clfs)]
            # print("Prediction score means: ",
            #       np.mean(np.mean(preds, 1)),
            #       np.std(np.mean(preds, 1)),
            #       np.mean(preds, 1))
            preds = [clf.predict_proba(np.expand_dims(latent[idx], 0))[0, 1] for idx, clf in zip(idxs, clfs)]
            print("Prediction score means: ",
                  np.mean(preds),
                  np.std(preds),)
            preds = np.mean(preds, 0)
            ac.append(preds)
        all_scores.append(ac)
        print()

    labels = ['Trained on $D_0$', 'Trained on $D_1$']
    for i, ac in enumerate(all_scores):
        # Brign down everything to fractions
        for x in ac:
            plot_x = x
            plt.hist(plot_x, 100, label=labels[i])

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    plt.legend()
    plt.title("Metal-classifier score prediction distributions for models on sample set")
    plt.xlabel("Meta-classifier $Pr$[trained on $D_1$]")
    plt.ylabel("Number of datapoints")
    plt.savefig("../visualize/score_distrs_celeba.png")

    # yeslabel = np.nonzero(all_stats[:, target_prop] == 1)[0]
    # nolabel  = np.nonzero(all_stats[:, target_prop] == 0)[0]

    # Pick relevant samples
    # label_attr     = attrs.index(inspect_these[1])
    # label_prop     = np.nonzero(all_stats[yeslabel, label_attr] == 1)[0]
    # label_noprop   = np.nonzero(all_stats[yeslabel, label_attr] == 0)[0]
    # nolabel_prop   = np.nonzero(all_stats[nolabel, label_attr] == 1)[0]
    # nolabel_noprop = np.nonzero(all_stats[nolabel, label_attr] == 0)[0]

    # all_cfms = np.array(all_cfms)

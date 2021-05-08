import utils
import numpy as np
import torch as ch
import torch.nn as nn
from sklearn.preprocessing import normalize
import kornia.augmentation as K
from torchvision.transforms import GaussianBlur, RandomGrayscale


def calibration(latents, weighted_align=True, use_ref=None):
    headstart = 0 if use_ref else 1

    perms = []
    # Take first latent as reference
    # Pass through examples
    for i in range(latents.shape[1]):
        p_perm = []
        # get order of first model as reference
        mapping = use_ref if use_ref else {
            x: kk for kk, x in enumerate(np.argsort(latents[0, i]))}
        for j in range(headstart, latents.shape[0]):
            p_perm.append([mapping.get(j) for j in np.argsort(latents[j, i])])
        perms.append(p_perm)
    perms = np.array(perms)

    # Weighted alignment
    weights = []
    if weighted_align:
        # For each model
        for i in range(perms.shape[1]):
            # For each dimension
            weighted = np.zeros((perms.shape[2], perms.shape[2]))
            for j in range(perms.shape[2]):
                u, c = np.unique(perms[:, i, j], return_counts=True)
                weighted[j][u] = c
            weights.append(weighted.T)

        weights = np.array(weights) / perms.shape[0]

    else:
        # For each model
        for i in range(perms.shape[1]):
            orders = []
            # Get frequencies across features
            for j in range(perms.shape[2]):
                unique, counts = np.unique(perms[:, i, j], return_counts=True)
                orders += [(u, c, j) for (u,c) in zip(unique, counts)]

            # Sort by increasing frequency counts
            orders = np.array(orders)
            orders = orders[np.argsort(-orders[:, 1])]

            weighted = np.zeros((perms.shape[2], perms.shape[2]))
            i = 0
            while i < orders.shape[0]:
                # If already-mapped index encountered, check for next
                while i < orders.shape[0] and np.sum(weighted[:, orders[i][0]]):
                    i += 1

                # If done traversing through everything, break out
                if i == orders.shape[0]:
                    break

                # Pick most-likely mapping (based on observation) across instances
                weighted[orders[i][2]][orders[i][0]] = 1
            # Check that 1:1 mapping is done
            assert np.sum(weighted) == perms.shape[2]

            # Store 1:1 mapping for this model
            weights.append(weighted)
        weights = np.array(weights)

    # Right-multiplty latent with weights to get aligned versions
    return mapping, weights


def lambdas(latents):
    mapwise_averages = np.mean(latents, axis=(-1, -2))
    # Get one lambda value per instance
    lambda_l = np.max(mapwise_averages, 1)
    # Normalize for comparison with other models
    # lambda_l /= np.max(lambda_l)
    return lambda_l


def balanced_cut(stats, attr_1, attr_2, sample_size):
    z_z = np.where(np.logical_and(stats[:, attr_1] == 0, stats[:, attr_2] == 0))[0]
    z_o = np.where(np.logical_and(stats[:, attr_1] == 0, stats[:, attr_2] == 1))[0]
    o_z = np.where(np.logical_and(stats[:, attr_1] == 1, stats[:, attr_2] == 0))[0]
    o_o = np.where(np.logical_and(stats[:, attr_1] == 1, stats[:, attr_2] == 1))[0]

    # Sample specified number per quadrant
    z_z = np.random.choice(z_z, sample_size)
    z_o = np.random.choice(z_o, sample_size)
    o_z = np.random.choice(o_z, sample_size)
    o_o = np.random.choice(o_o, sample_size)

    # Combine all these indices
    combined = np.concatenate((z_z, z_o, o_z, o_o))
    assert (combined.shape[0] == 4 * sample_size), "Required sample-size not achieved!"
    return combined


def get_predictions(model, x, batch_size):
    i = 0
    preds = []
    for i in range(0, x.shape[0], batch_size):
        x_ = x[i:i+batch_size]
        y_ = model(x_.cuda()).detach().cpu()
        preds.append(y_)
    return ch.cat(preds)


def augmentation_robustness(x,
                            deg=0,
                            jitter_tup=(0, 0, 0, 0),
                            translate=(0, 0),
                            erase_scale=(0, 0),
                            grayscale=False):
    # Shift to [0, 1] space
    x_ = (x * 0.5) + 0.5
    transforms = []
    if erase_scale[1] > 0:
        transforms.append(K.RandomErasing(scale=erase_scale,
                                          p=1))

    if translate[0] > 0 or translate[1] > 0 or deg > 0:
        transforms.append(K.RandomAffine(degrees=deg,
                                         translate=translate,
                                         p=1))

    # transforms.append(K.ColorJitter(*jitter_tup))
    if jitter_tup[-1] > 0:
        transforms.append(GaussianBlur(kernel_size=19,
                                       sigma=jitter_tup[-1]))

    # Grayscale-based check
    if grayscale:
        transforms.ppend(RandomGrayscale(p=1))
    transform = nn.Sequential(*transforms)

    augmented = transform(x_)
    # Shift back to [-1, 1] space
    return (augmented - 0.5) / 0.5


def collect_augmented_data(loader,
                           deg=0,
                           jitter=(0, 0, 0, 0),
                           translate=(0, 0),
                           erase_scale=(0, 0)):
    X, X_aug, Y = [], [], []
    for x, y in loader:
        X_aug.append(augmentation_robustness(x,
                                             deg=deg,
                                             jitter_tup=jitter,
                                             translate=translate,
                                             erase_scale=erase_scale))
        X.append(x)
        Y.append(y)
    return (X, X_aug, Y)


def get_robustness_shifts(model_fn, augdata, target, prop_id):
    counts = [0, 0]
    noprop, prop = [0, 0], [0, 0]
    for x, x_adv, y in zip(*augdata):
        y_picked = y[:, target].cuda()
        y_t = y[:, prop_id].numpy()

        prop_idex = np.nonzero(y_t == 1)[0]
        noprop_idex = np.nonzero(y_t == 0)[0]

        before_preds = ((model_fn(x.cuda()) >= 0) == y_picked).cpu()
        after_preds = ((model_fn(x_adv) >= 0) == y_picked).cpu()

        noprop[0] += ch.sum(1.0 * before_preds[noprop_idex]).item()
        noprop[1] += ch.sum(1.0 * after_preds[noprop_idex]).item()

        prop[0] += ch.sum(1.0 * before_preds[prop_idex]).item()
        prop[1] += ch.sum(1.0 * after_preds[prop_idex]).item()

        counts[0] += prop_idex.shape[0]
        counts[1] += noprop_idex.shape[0]

    for i in range(2):
        prop[i] /= counts[0]
        noprop[i] /= counts[1]

    return noprop, prop

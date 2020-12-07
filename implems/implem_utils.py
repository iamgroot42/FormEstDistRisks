import utils
import numpy as np
import torch as ch
import torch.nn as nn

from collections import OrderedDict
from aif360.metrics import ClassificationMetric


def get_latents(mainmodel, dataloader, method_type):
    all_stats = []
    all_latent = []
    for (x, y) in dataloader:

        # Not-last latent features needed
        if method_type < 0:
            latent = mainmodel(x.cuda(),
                               deep_latent=-method_type).detach()
            # latent = latent.view(latent.shape[0], -1)
        # Latent features needed
        elif method_type == 0 or method_type == 1 or method_type == 4:
            latent = mainmodel(x.cuda(), only_latent=True).detach()
        # Use logit scores
        elif method_type == 2:
            latent = mainmodel(x.cuda()).detach()[:, 0]
        # Use post-sigmoid outputs
        elif method_type == 3:
            latent = ch.sigmoid(mainmodel(x.cuda()).detach()[:, 0])

        all_latent.append(latent.cpu().numpy())
        all_stats.append(y.cpu().numpy())

    all_latent = np.concatenate(all_latent, 0)
    all_stats = np.concatenate(all_stats)

    # Normalize according to max entry?
    all_latent /= np.max(all_latent, 1, keepdims=True)
    # [0, 1] scaling
    # max_l, min_l = np.max(all_latent, 1, keepdims=True), np.min(all_latent, 1, keepdims=True)
    # all_latent = (all_latent - min_l) / (max_l - min_l)

    return all_latent, all_stats


def calibration(latents, weighted_align=True, use_ref=None):
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


def get_features_for_model(dataloader, MODELPATH, weight_init,
                           method_type, layers=[64, 16]):
    # Load model
    model = utils.FaceModel(512,
                            train_feat=True,
                            weight_init=weight_init,
                            hidden=layers).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(MODELPATH), strict=False)
    model.eval()

    # Get latent representations
    lat, sta = get_latents(model, dataloader, method_type)
    # lat = np.sort(lat, 1)
    # lat = np.array([np.std(lat, 1), np.mean(lat == 0, 1), np.mean(lat, 1), np.mean(lat ** 2, 1)]).T
    return (lat, sta)


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


def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                  dataset_pred, 
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    metrics["False discovery rate difference"] = classified_metric_pred.false_discovery_rate_difference()
    metrics["False omission rate difference"] = classified_metric_pred.false_omission_rate_difference()

    return metrics

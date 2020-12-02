import numpy as np
import torch as ch


def get_latents(mainmodel, dataloader, method_type):
    all_stats = []
    all_latent = []
    for (x, y) in dataloader:

        # latent = mainmodel(x.cuda(), deep_latent=6).detach()
        # latent = latent.view(latent.shape[0], -1)

        # Latent features needed
        if method_type == 0 or method_type == 1 or method_type == 4:
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
    max_l, min_l = np.max(all_latent, 1, keepdims=True), np.min(all_latent, 1, keepdims=True)
    # [0, 1] scaling
    all_latent = (all_latent - min_l) / (max_l - min_l)
    # all_latent /= np.max(all_latent, 1, keepdims=True)

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
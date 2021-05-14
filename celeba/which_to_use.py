import numpy as np
import utils
from PIL import Image
from tqdm import tqdm
import os
from facenet_pytorch import MTCNN


def get_labels(dataloader):
    labels = []
    for (_, y) in tqdm(dataloader, total=len(dataloader)):
        labels.append(y)
    return np.concatenate(labels)


def filter(indices, value, ratio):
    qi = np.nonzero(indices == value)[0]
    notqualify = np.nonzero(indices != value)[0]
    np.random.shuffle(notqualify)
    nqi = notqualify[:int(((1-ratio) * len(qi))/ratio)]
    return np.sort(np.concatenate((qi, nqi)))


def balanced_split(labels, care_about, split_ratio):
    clusters = {}
    split_1, split_2 = [], []
    for i, lb in enumerate(labels):
        key = "".join(str(c) for c in lb[care_about])
        current = clusters.get(key, [])
        current.append(i)
        clusters[key] = current

    for x in clusters.values():
        np.random.shuffle(x)
        split_point = int(split_ratio * len(x))
        split_1.append(x[:split_point])
        split_2.append(x[split_point:])

    split_1 = np.concatenate(split_1)
    split_2 = np.concatenate(split_2)
    return [split_1, split_2]


def dump_files(path, model, dataloader, indices, target_prop):
    # Make directories for classes (binary)
    os.mkdir(os.path.join(path, "0"))
    os.mkdir(os.path.join(path, "1"))
    trace, start = 0, 0
    done_with_loader = False
    for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y_ = y[:, target_prop]

        cropped_faces, indices_ = utils.get_cropped_faces(model, x)
        cropped_faces = cropped_faces.numpy()
        # Shift back to [0, 1]
        cropped_faces = (cropped_faces * 0.5) + 0.5

        for idx, j in enumerate(indices_):
            # If current index not found
            # Skip until you're not above and beyond
            if start + j > indices[trace]:
                trace += 1
                # If run through all indices there are in data, stop processing
                if trace == len(indices):
                    done_with_loader = True
                    break
                continue
            if start + j == indices[trace]:
                image = Image.fromarray(
                    (255 * np.transpose(cropped_faces[idx], (1, 2, 0))).astype('uint8'))
                file_prefix = "".join(str(c) for c in y[j].numpy())
                if y_[j] == 0:
                    image.save(os.path.join(
                        path, "0", file_prefix + "_" + str(trace)) + ".png")
                else:
                    image.save(os.path.join(
                        path, "1", file_prefix + "_" + str(trace)) + ".png")

                trace += 1
                # If run through all indices there are in data, stop processing
                if trace == len(indices):
                    done_with_loader = True
                    break

        start += y.shape[0]
        if done_with_loader:
            break


if __name__ == "__main__":
    import sys
    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    target_prop = attrs.index("Smiling")
    print(attrs)

    trainloader, testloader = ds.make_loaders(
            batch_size=256, workers=8, shuffle_train=False, shuffle_val=False)
    labels_tr = get_labels(trainloader)
    labels_te = get_labels(testloader)

    savemode = False
    indices_basepath = sys.argv[1]
    care_about = ['Attractive', 'Male', 'Young', 'Smiling']
    care_about = [attrs.index(x) for x in care_about]

    if savemode:
        first_split_ratio = float(sys.argv[2])    
        # Perform first time, load from memory next time
        splits_tr = balanced_split(labels_tr, care_about, first_split_ratio)
        splits_te = balanced_split(labels_te, care_about, first_split_ratio)

        # Save these splits
        np.save(os.path.join(indices_basepath, "splits_tr_1"), splits_tr[0])
        np.save(os.path.join(indices_basepath, "splits_tr_2"), splits_tr[1])
        np.save(os.path.join(indices_basepath, "splits_te_1"), splits_te[0])
        np.save(os.path.join(indices_basepath, "splits_te_2"), splits_te[1])
        print("Dumped indices into numpy arrays!")

    else:
        # Save processed images here
        dataset_basepath = sys.argv[2]
        given_prop = sys.argv[3]
        ratio_wanted = float(sys.argv[4])
        assert ratio_wanted <= 1 and ratio_wanted >= 0, "Provide valid ratio in [0, 1]"

        # 0.68 for attractive, 0.59 for male, 0.37 for young

        # Use 23000 per class for train
        # And 3000 per class for test
        num_per_class_train = 23000
        num_per_class_test = 3000

        # Load from memory
        splits_tr = [np.load(os.path.join(indices_basepath,
                                          "splits_tr_1.npy")),
                     np.load(os.path.join(indices_basepath,
                                          "splits_tr_2.npy"))]
        splits_te = [np.load(os.path.join(indices_basepath,
                                          "splits_te_1.npy")),
                     np.load(os.path.join(indices_basepath,
                                          "splits_te_2.npy"))]
        print("Loaded split information from memory")

        # Write these splits to numpy file and read later
        # To maintain consistency for shared/non-shared data

        # Load cropping model
        model = MTCNN(device='cuda')
        paths = [
            os.path.join(dataset_basepath, "split_1"),
            os.path.join(dataset_basepath, "split_2")
        ]

        for i in range(2):
            print("Original label balance:", np.mean(
                labels_tr[splits_tr[i], target_prop]))

            # Class labels
            clabels_tr = labels_tr[splits_tr[i], target_prop]
            clabels_te = labels_te[splits_te[i], target_prop]

            if given_prop == "none":
                # No filter (all data)
                picked_indices_tr = np.arange(splits_tr[i].shape[0])
                picked_indices_te = np.arange(splits_te[i].shape[0])

                # Heuristic: generate samples, pick the one that is closest
                # In terms of preserving ratios that we care about
                # Across all experiments
                def heuristic(idcs, lbls, gt_labels,
                              cwise_sample, n_tries=1000):
                    # Get original ratio values
                    og_vals = np.array(
                        [np.mean(lbls[idcs, co]) for co in care_about])
                    vals, pckds = [], []
                    iterator = tqdm(range(n_tries))
                    cur_best_val = np.inf
                    for _ in iterator:
                        # Class-balanced sampling
                        zero_ids = np.nonzero(gt_labels == 0)[0]
                        one_ids = np.nonzero(gt_labels == 1)[0]
                        zero_ids = np.random.permutation(
                            zero_ids)[:cwise_sample]
                        one_ids = np.random.permutation(
                            one_ids)[:cwise_sample]
                        # Combine them together
                        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
                        vals.append(
                            np.array([np.mean(
                                lbls[pckd, co]) for co in care_about]))
                        pckds.append(pckd)

                        cur_best_val = np.minimum(
                            cur_best_val, np.sum((vals[-1] - og_vals)**2))

                        # Print best ratio so far in descripton
                        iterator.set_description(
                            "Ratios L-2 dist: %.4f" % cur_best_val)

                    vals = [np.sum((x - og_vals)**2) for x in vals]
                    # Pick the one closest to desired ratio
                    return pckds[np.argmin(vals)]

                picked_indices_tr = heuristic(
                    picked_indices_tr, labels_tr,
                    clabels_tr, num_per_class_train)
                picked_indices_te = heuristic(
                    picked_indices_te, labels_te,
                    clabels_te, num_per_class_test)
            else:
                prop = attrs.index(given_prop)
                tags_tr = labels_tr[splits_tr[i], prop]
                tags_te = labels_te[splits_te[i], prop]

                print("Original property ratio:", np.mean(tags_tr))
                # At this stage, care only about label
                # Balance and preserving property

                def heuristic(tgs, foc, fil, gt_labels,
                              cwise_sample, n_tries=1000):
                    vals, pckds = [], []
                    fill_comp = fil
                    if foc == 0:
                        fill_comp = 1 - fill_comp
                    iterator = tqdm(range(n_tries))
                    for _ in iterator:
                        pckd = filter(tgs, foc, fil)
                        # Class-balanced sampling
                        zero_ids = np.nonzero(gt_labels[pckd] == 0)[0]
                        one_ids = np.nonzero(gt_labels[pckd] == 1)[0]
                        zero_ids = np.random.permutation(
                            pckd[zero_ids])[:cwise_sample]
                        one_ids = np.random.permutation(
                            pckd[one_ids])[:cwise_sample]
                        # Combine them together
                        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
                        vals.append(np.mean(tgs[pckd]))
                        pckds.append(pckd)

                        # Print best ratio so far in descripton
                        iterator.set_description(
                            "%.4f" % (fill_comp + np.min([np.abs(zz-fill_comp) for zz in vals])))

                    vals = np.abs(np.array(vals) - fill_comp)
                    # Pick the one closest to desired ratio
                    return pckds[np.argmin(vals)]

                if given_prop == "Attractive":
                    # Attractive
                    picked_indices_tr = heuristic(
                        tags_tr, 1, ratio_wanted,
                        clabels_tr, num_per_class_train)
                    picked_indices_te = heuristic(
                        tags_te, 1, ratio_wanted,
                        clabels_te, num_per_class_test)
                elif given_prop == "Male":
                    # Male
                    picked_indices_tr = heuristic(
                        tags_tr, 1, ratio_wanted,
                        clabels_tr, num_per_class_train)
                    picked_indices_te = heuristic(
                        tags_te, 1, ratio_wanted,
                        clabels_te, num_per_class_test)
                elif given_prop == "Young":
                    # Old
                    picked_indices_tr = heuristic(
                        tags_tr, 0, ratio_wanted,
                        clabels_tr, num_per_class_train)
                    picked_indices_te = heuristic(
                        tags_te, 0, ratio_wanted,
                        clabels_te, num_per_class_test)
                else:
                    raise ValueError("Ratio for this property not defined yet")

                print("Filtered property ratio:",
                      np.mean(tags_tr[picked_indices_tr]))

            actual_picked_indices_tr = sorted(
                splits_tr[i][picked_indices_tr])
            actual_picked_indices_te = sorted(
                splits_te[i][picked_indices_te])

            print("Filtered data label balance:", np.mean(
                labels_tr[actual_picked_indices_tr, target_prop]))

            # Get loaders again
            trainloader, testloader = ds.make_loaders(
                batch_size=512, workers=4, shuffle_train=False,
                shuffle_val=False, data_aug=False)
            # Save test data
            os.mkdir(os.path.join(paths[i],   "test"))
            dump_files(os.path.join(paths[i], "test"),  model,
                       testloader,  actual_picked_indices_te, target_prop)

            # Get loaders again
            trainloader, testloader = ds.make_loaders(
                batch_size=512, workers=4, shuffle_train=False,
                shuffle_val=False, data_aug=False)
            # Save train data
            os.mkdir(os.path.join(paths[i],   "train"))
            dump_files(os.path.join(paths[i], "train"), model,
                       trainloader, actual_picked_indices_tr, target_prop)

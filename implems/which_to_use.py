import numpy as np
import utils
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import os

from facenet_pytorch import InceptionResnetV1, MTCNN


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
    for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y_ = y[:, target_prop]

        cropped_faces, indices_ = utils.get_cropped_faces(model, x)
        cropped_faces = cropped_faces.numpy()
        # Shift back to [0, 1]
        cropped_faces = (cropped_faces * 0.5) + 0.5

        for idx, j in enumerate(indices_):
            # If current index not found, skip until you're not above and beyond
            if start + j > indices[trace]:
                trace += 1
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
                    return

        start += y.shape[0]

        # for j in range(y_.shape[0]):
        # 	if start + j == indices[trace]:
        # 		if y_[j] == 0:
        # 			image = Image.fromarray((255 * np.transpose(x_[j], (1, 2, 0))).astype('uint8'))
        # 			try:
        # 				model(image, save_path=os.path.join(path, "0", str(trace)) + ".png")
        # 			except:
        # 				# Side view, no face to look at
        # 				print("Class 0: Problematic image!")
        # 		else:
        # 			image = Image.fromarray((255 * np.transpose(x_[j], (1, 2, 0))).astype('uint8'))
        # 			try:
        # 				model(image, save_path=os.path.join(path, "1", str(trace)) + ".png")
        # 			except:
        # 				# Side view, no face to look at
        # 				print("Class 1: Problematic image!")
        # 			# image.save(os.path.join(path, "1", str(trace)) + ".png")
        # 		trace += 1
        # 		# If run through all indices there are in data, stop processing
        # 		if trace == len(indices): return
        # start += y.shape[0]


if __name__ == "__main__":

    constants = utils.Celeb()
    ds = constants.get_dataset()

    trainloader, testloader = ds.make_loaders(
        batch_size=256, workers=8, shuffle_train=False, shuffle_val=False)
    attrs = constants.attr_names
    print(attrs)

    care_about = ['Attractive', 'Male', 'Young', 'Smiling']
    care_about = [attrs.index(x) for x in care_about]
    # prop = attrs.index("Male")
    # prop = attrs.index("Young")

    target_prop = attrs.index("Smiling")
    # target_prop = attrs.index("Male")
    labels_tr = get_labels(trainloader)
    labels_te = get_labels(testloader)

    # Perform first time, load from memory next time
    # splits_tr = balanced_split(labels_tr, care_about, 0.7)
    # splits_te = balanced_split(labels_te, care_about, 0.7)

    # Save these splits
    # np.save("splits_tr_1", splits_tr[0])
    # np.save("splits_tr_2", splits_tr[1])
    # np.save("splits_te_1", splits_te[0])
    # np.save("splits_te_2", splits_te[1])
    # print("Dumped indices into numpy arrays!")
    # exit(0)

    # Load from memory
    basepath = "/u/as9rw/work/fnb/implems/"
    splits_tr = [np.load(basepath + "splits_tr_1.npy"),
                 np.load(basepath + "splits_tr_2.npy")]
    splits_te = [np.load(basepath + "splits_te_1.npy"),
                 np.load(basepath + "splits_te_2.npy")]
    print("Loaded split information from memory")

    # Write these splits to numpy file and read later
    # To maintain consistency for shared/non-shared data

    # Load cropping model
    model = MTCNN(device='cuda')
    paths = [
        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/attractive/split_1",
        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/attractive/split_2"

        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_1",
        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2"

        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/male/split_1",
        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/male/split_2"

        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/old/split_1",
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/old/split_2"
    ]

    for i in range(2):

        # print("Original label balance:", np.mean(labels_tr[:, target_prop]))
        print("Original label balance:", np.mean(
            labels_tr[splits_tr[i], target_prop]))

        # tags_tr = labels_tr[:, prop]
        # tags_te = labels_te[:, prop]

        prop = attrs.index("Young")
        tags_tr = labels_tr[splits_tr[i], prop]
        tags_te = labels_te[splits_te[i], prop]

        print("Original property ratio:", np.mean(tags_tr))

        # No filter (all data)
        # picked_indices_tr = np.arange(labels_tr.shape[0])
        # picked_indices_te = np.arange(labels_te.shape[0])

        # No filter (all data)
        # picked_indices_tr = np.arange(splits_tr[i].shape[0])
        # picked_indices_te = np.arange(splits_te[i].shape[0])
        # Attractive
        # picked_indices_tr = filter(tags_tr, 1, 0.68)
        # picked_indices_te = filter(tags_te, 1, 0.68)
        # Male
        # picked_indices_tr = filter(tags_tr, 1, 0.59)
        # picked_indices_te = filter(tags_te, 1, 0.59)
        # Old
        picked_indices_tr = filter(tags_tr, 0, 0.37)
        picked_indices_te = filter(tags_te, 0, 0.37)
        print("Filtered property ratio:", np.mean(tags_tr[picked_indices_tr]))
        actual_picked_indices_tr = sorted(splits_tr[i][picked_indices_tr])
        actual_picked_indices_te = sorted(splits_te[i][picked_indices_te])
        print("Filtered data label balance:", np.mean(
            labels_tr[actual_picked_indices_tr, target_prop]))

        # print("Filtered data label balance:", np.mean(labels_tr[picked_indices_tr, target_prop]))

        # Get loaders again
        trainloader, testloader = ds.make_loaders(
            batch_size=512, workers=8, shuffle_train=False, shuffle_val=False, data_aug=False)
        # Save test data
        os.mkdir(os.path.join(paths[i],   "test"))
        dump_files(os.path.join(paths[i], "test"),  model,
                   testloader,  actual_picked_indices_te, target_prop)
        # dump_files(os.path.join(paths[i], "test"),  model, testloader,  picked_indices_te, target_prop)
        # Save train data
        os.mkdir(os.path.join(paths[i],   "train"))
        dump_files(os.path.join(paths[i], "train"), model,
                   trainloader, actual_picked_indices_tr, target_prop)
        # dump_files(os.path.join(path[i], "train"), model, trainloader, picked_indices_tr, target_prop)

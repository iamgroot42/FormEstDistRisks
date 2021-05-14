from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from robustness.tools.vis_tools import show_image_row
import numpy as np
import torch as ch
from tqdm import tqdm
from scipy.spatial import distance

import utils


def get_match_scores(source_image, target_images, featurizer, model):
	scores = []
	if featurizer == "cosine":
		with ch.no_grad():
			si = ch.from_numpy(np.array([source_image]).transpose(0, 3, 1, 2)).cuda()
			ti = ch.from_numpy(target_images.transpose(0, 3, 1, 2)).cuda()
			source_latent, _ = model(si, with_latent=True, fake_relu=False, just_latent=True)
			target_latents = []
			bs = 1000
			for i in range(0, len(ti), bs):
				target_latent, _ = model(ti[i: i + bs], with_latent=True, fake_relu=False, just_latent=True)
				target_latents.append(target_latent)
			target_latents = ch.cat(target_latents, 0)
		target_latents = target_latents.cpu().numpy()
		source_latent = source_latent[0].cpu().numpy()
		for t in tqdm(target_latents):
			scores.append(1 - distance.cosine(source_latent, t))
	else:
		for t in tqdm(target_images):
			# Make sure last dimension is channels
			scores.append(featurizer(t, source_image, data_range=1., multichannel=True))
	return scores


if __name__ == "__main__":
	ftype = "cosine"
	# ftype = "ssim"
	if ftype == "ssim":
		featurizer = ssim
	elif ftype == "cosine":
		featurizer = ftype
	else:
		raise ValueError("Feature Extractor not supported")
	
	# Get training-set images and model
	this_to_that_map = [2, 1, 3, 4, 5, 6, 7, 0, 8, 9]
	# constants = utils.RobustCIFAR10("/p/adversarialml/as9rw/generated_images_notgray_filtered_2", None)
	# constants = utils.RobustCIFAR10("/p/adversarialml/as9rw/generated_images_notgray", None)
	# constants = utils.RobustCIFAR10("/p/adversarialml/as9rw/generated_images", None)
	constants = utils.RobustCIFAR10("/p/adversarialml/as9rw/generated_images_filtered", None)
	cifar_constant = utils.CIFAR10()
	model = cifar_constant.get_model("linf" , "vgg19")
	
	# Get dataset
	ds = constants.get_dataset()
	_, data_loader = ds.make_loaders(batch_size=500, workers=8, shuffle_train=False, shuffle_val=False, data_aug=False, only_val=True)
	train_images = []
	# bird  car  cat  deer  dog  frog  horse  plane  ship  truck
	target_label = 6
	for (im, label) in data_loader:
		train_images.append(im[label == target_label].numpy())
	train_images = np.concatenate(train_images, 0)
	train_images = train_images.transpose(0, 2, 3, 1)

	# Go through images, save in new directory images with label-probability above a certain threshold
	# score_threshold = 0.8
	# glob_counter = 1
	# import os
	# save_path = "/p/adversarialml/as9rw/generated_images_notgray_filtered_2/test"
	# for (im, label) in tqdm(data_loader):
	# 	scores, _ = model(im.cuda())
	# 	scores = ch.nn.Softmax(dim=1)(scores)
	# 	keep_indices = ["bird",  "car",  "cat",  "deer",  "dog",  "frog",  "horse",  "plane",  "ship",  "truck"]
	# 	# Retain images only with specific thresold prediction scores
	# 	for i in range(label.shape[0]):
	# 		if scores[i][this_to_that_map[label[i]]] >= score_threshold:
	# 			# Save image in folder accordingly
	# 			from PIL import Image
	# 			im_ = Image.fromarray((im[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
	# 			im_.save(os.path.join(save_path, keep_indices[label[i]], str(glob_counter) + ".png"))
	# 			glob_counter += 1
	# print("Done!")
	# exit(0)

	# Dummy source image
	print("%d examples present" % train_images.shape[0])
	# sc_img = train_images[9999]

	best_images = []
	clusters = []
	threshold = 2#8#2
	indices = np.arange(train_images.shape[0])
	while len(indices) > 0:
		sc_img = train_images[indices[0]]
		best_images.append(sc_img)
		l2_distances = np.linalg.norm((train_images[indices] - sc_img).reshape((len(indices), -1)), ord=2, axis=1)
		qualified_matches = (l2_distances <= threshold)
		clusters.append(indices[qualified_matches])
		indices = indices[~qualified_matches]
		print("%d size cluster, %d remain" % (np.sum(qualified_matches), len(indices)))
		if len(clusters) >= 25:
			break

	print("%d seemingly unique images identified for threshold %.2f" % (len(clusters), threshold))

	# Load image for which nearest neighbor is to be searched

	# Visualize top-N matches
	# top_k = 256
	# segment = 16
	# top_k = 200
	# segment = 2
	# sort_indices = np.argsort(-np.array(scores))[top_k * (segm1nt-1):top_k * segment]
	# best_images = train_images[sort_indices]

	# # dist_l2   = ch.norm(flatten, p=2, dim=-1)
	# l2_distances = np.linalg.norm((best_images - sc_img).reshape((best_images.shape[0], -1)), ord=2, axis=1)
	# print(l2_distances)

	# # 8 bit space
	# sc_img = ((255 * sc_img).astype('uint8') / 255).astype('float32')

	# best_images = np.concatenate([np.array([sc_img]), best_images], 0)
	best_images = np.array(best_images)
	# Sort by average color/illumination
	best_images = best_images[:200]


	# Save specific image
	np.save("./visualize/closest_to_this", best_images[0])

	best_images = ch.from_numpy(best_images.transpose(0, 3, 1, 2))
	# Get scores for specific class for these images
	with ch.no_grad():
		scores, _ = model(best_images.cuda())
		scores = ch.nn.Softmax(dim=1)(scores)
		# bird  car  cat  deer  dog  frog  horse  plane  ship  truck
		# mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		lab_scores = ["%.3f" % x.item() for x in scores[:, this_to_that_map[target_label]]]

	show_image_row([best_images], ["Unique Images"], fontsize=22,
				tlist=[lab_scores],
				filename="./visualize/cluster_identifiers.png")


	# show_image_row([best_images],
	# 			["Closest Matches"],
	# 			fontsize=22,
	# 			filename="./visualize/genmatches_%s.png" % ftype)

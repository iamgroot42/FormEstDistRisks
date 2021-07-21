import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from robustness.tools.vis_tools import show_image_row
import numpy as np
import torch as ch
from tqdm import tqdm
from scipy.spatial import distance

import utils


# img = cv2.imread('simple.jpg',0)


def get_match_scores(source_image, target_images, featurizer, uses_cv2, model):
	scores = []
	if uses_cv2:
		# Prepare matches
		FLANN_INDEX_KDTREE = 0
		index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)

		flann = cv2.FlannBasedMatcher(index_params, search_params)

		# Precompute features
		source_feature  = featurizer.detectAndCompute(cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR), None)

		# for attribute in dir(featurizer):
		# 	if not attribute.startswith("get"):
		# 		continue
		# 	param = attribute.replace("get", "")
		# 	get_param = getattr(featurizer, attribute)
		# 	val = get_param()
		# 	print(param, '=', val)

		# exit(0)
		target_features = []
		for t in tqdm(target_images):
			target_features.append(featurizer.detectAndCompute(cv2.cvtColor(t, cv2.COLOR_RGB2BGR), None))
		for tf in tqdm(target_features):
			if tf[1] is None or len(tf[1]) < 2:
				scores.append(0)
				continue

			matches = flann.knnMatch(np.float32(source_feature[1]), np.float32(tf[1]), k=2)
			number_keypoints = min(len(source_feature[0]), len(tf[0]))
			good = []
			best_percent = 0
			for m,n in matches:
				if m.distance < 0.75*n.distance:
					good.append([m])
					percent=(len(good)*100)/number_keypoints
					best_percent = max(best_percent, percent)

			scores.append(best_percent)
			# exit(0)
	else:
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
	import sys
	dataset = sys.argv[1]
	uses_cv2 = True
	# ftype = "orb"
	ftype = "cosine"
	# ftype = "ssim"
	if ftype == "orb":
		ps = 7
		# featurizer = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE, edgeThreshold=ps, patchSize=7)
		featurizer = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_HARRIS_SCORE, edgeThreshold=ps, patchSize=7)
	# elif ftype == "sift":
		# featurizer = cv2.SIFT_create()
	# elif ftype == "surf":
		# featurizer = cv2.SURF_create()
	elif ftype == "ssim":
		featurizer = ssim
		uses_cv2 = False
	elif ftype == "cosine":
		featurizer = ftype
		uses_cv2 = False
	else:
		raise ValueError("Feature Extractor not supported")
	
	# Get training-set images and model
	if dataset == "cifar":
		constants = utils.CIFAR10()
		model = constants.get_model("linf" , "vgg19")
	else:
		constants = utils.SVHN10()
		model = constants.get_model("linf" , "vgg16")
	# Get dataset
	ds = constants.get_dataset()
	train_loader, _ = ds.make_loaders(batch_size=250, workers=8, shuffle_train=False, shuffle_val=False, data_aug=False)
	train_images = []
	target_label = 1
	#0 plane, 1 car, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
	for (im, label) in train_loader:
		train_images.append(im[label == target_label].numpy())
		# train_images.append(im.numpy())
	train_images = np.concatenate(train_images, 0)
	train_images = train_images.transpose(0, 2, 3, 1)

	# Dummy source image
	sc_img = np.load("./visualize/closest_to_this.npy")
	# sc_img = train_images[0]

	# sc_img *= 0.9
	# sc_img = 1 - sc_img

	# Load image for which nearest neighbor is to be searched
	scores = get_match_scores(sc_img, train_images, featurizer, uses_cv2, model)

	# Visualize top-N matches
	# top_k = 256
	# segment = 16
	top_k = 128
	segment = 3#1
	sort_indices = np.argsort(-np.array(scores))[top_k * (segment-1):top_k * segment]
	best_images = train_images[sort_indices]
	# Add original images to matches for visualization reference
	# print(best_images.shape)
	samples_min = best_images.T.reshape(3, -1).min(1)
	samples_max = best_images.T.reshape(3, -1).max(1)
	# for i in range(3):
	# 	sc_img[:,:,i] *= (samples_max[i] - samples_min[i])
	# 	sc_img[:,:,i] += samples_min[i]

	# 8 bit space
	sc_img = ((255 * sc_img).astype('uint8') / 255).astype('float32')


	best_images = np.concatenate([np.array([sc_img]), best_images], 0)
	best_images = ch.from_numpy(best_images.transpose(0, 3, 1, 2))

	show_image_row([best_images],
				["Closest Matches"],
				fontsize=22,
				filename="./visualize/%s.png" % ftype)
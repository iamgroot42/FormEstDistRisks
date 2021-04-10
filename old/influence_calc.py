from robustness.tools.vis_tools import show_image_row
import utils
import pytorch_influence_functions as ptif
import numpy as np
import torch as ch
from pathlib import Path


if __name__ == "__main__":
	# Ready data utils
	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	# Load model
	model = constants.get_model("linf" , "vgg19")
	wrapped_model = utils.MadryToNormal(model)
	# Load target test image
	sc_img = np.load("./visualize/closest_to_this.npy")

	#0 plane, 1 car, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck

	sc_img_ch = ch.from_numpy(np.expand_dims(sc_img.transpose(2, 0, 1), 0))
	label = ch.from_numpy(np.array([9]))
	# # Ready data loaders
	batch_size=4
	train_loader, _ = ds.make_loaders(batch_size=batch_size, workers=8, fixed_train_order=False, data_aug=False)
	s_test_vec = ptif.calc_s_test_single(wrapped_model, sc_img_ch, label, train_loader, gpu=0, r=2, scale=5e2, recursion_depth=5000)

	# Reset train loader
	train_loader, _ = ds.make_loaders(batch_size=batch_size, workers=8, fixed_train_order=True, data_aug=False)
	influences, harmful, helpful  = ptif.calc_influence_single(wrapped_model, train_loader, 0, None, None, s_test_vec=s_test_vec)

	print(helpful[:64])
	print(harmful[:64])

	# helpful = np.array([5074, 22490, 43089, 36288, 11734, 15696, 33520, 10989, 17753, 37694, 19569, 27702, 25549, 15891, 39877, 21768, 25423, 45674, 15466, 47318, 5022, 49942, 25977, 45987, 45507, 1055, 33244, 19554, 13031, 32047, 24461, 25466])
	# harmful = np.array([34023, 24256, 19052, 49447, 23673, 29119, 11270, 17590, 36622, 16208, 14275, 19147, 18310, 39362, 13557, 13424, 28124, 40148, 33811, 37219, 39477, 42025, 43476, 23618, 18977, 40350, 18857, 18924, 24570, 41157, 1676, 36928])

	# Reset train loader
	train_loader, _ = ds.make_loaders(batch_size=512, workers=8, fixed_train_order=True, data_aug=False)
	all_images, all_labels = utils.load_all_loader_data(train_loader)

	helpful_images = all_images[helpful[:64]].numpy()
	harmful_images = all_images[harmful[:64]].numpy()

	helpful_images = helpful_images.transpose(0, 2, 3, 1)
	harmful_images = harmful_images.transpose(0, 2, 3, 1)

	# 8 bit space
	sc_img = ((255 * sc_img).astype('uint8') / 255).astype('float32')

	helpful_images = np.concatenate([np.array([sc_img]), helpful_images], 0)
	helpful_images = ch.from_numpy(helpful_images.transpose(0, 3, 1, 2))
	harmful_images = np.concatenate([np.array([sc_img]), harmful_images], 0)
	harmful_images = ch.from_numpy(harmful_images.transpose(0, 3, 1, 2))

	show_image_row([helpful_images, harmful_images],
				["Helpful", "Harmful"],
				fontsize=22,
				filename="./visualize/influence.png")

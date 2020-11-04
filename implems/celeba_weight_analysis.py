import numpy as np
import utils
import torch as ch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_latents(mainmodel, dataloader):
	all_stats = []
	all_latent = []
	for (x, y) in dataloader:

		# latent = mainmodel(x.cuda(), deep_latent=14).detach()
		# latent = mainmodel(x.cuda(), deep_latent=5).detach()
		# latent = latent.view(latent.shape[0], -1)

		latent = mainmodel(x.cuda(), only_latent=True).detach()
		all_latent.append(latent.cpu().numpy())
		all_stats.append(y.cpu().numpy())

	all_latent = np.concatenate(all_latent, 0)
	all_stats  = np.concatenate(all_stats)

	return all_latent, all_stats


def get_features_for_model(dataloader, MODELPATH):
	# Load model
	model = utils.FaceModel(512, train_feat=True).cuda()
	model = nn.DataParallel(model)
	model.load_state_dict(ch.load(MODELPATH))
	model.eval()

	# Get latent representations
	return get_latents(model, dataloader)


if __name__ == "__main__":

	constants = utils.Celeb()
	ds = constants.get_dataset()

	attrs = constants.attr_names
	inspect_these = ["Attractive", "Male", "Young"]

	folder_paths = [
					"celeba_models/more/smile_crop_aug/all",
					# "celeba_models/more/smile_crop_aug/attractive",
					"celeba_models/more/smile_crop_aug/male",
					# "celeba_models/more/smile_crop_aug/old"
					]
	
	blind_test_models = [
		"/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/10_0.9289294306335204",
		"/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/9_0.926172814755413",
		"/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/8_0.9238171611868484",

		"/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs/10_0.8991738066095472",
		"/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs/9_0.9121022031823746",
		"/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs/7_0.9226591187270502"
	]

	cropmodel = MTCNN(device='cuda')

	# Get all cropped images
	x_cropped, y_cropped = [], []
	_, dataloader = ds.make_loaders(batch_size=512, workers=8, shuffle_train=False, shuffle_val=False, only_val=True)
	for x, y in tqdm(dataloader, total=len(dataloader)):
		x_, indices = utils.get_cropped_faces(cropmodel, x)
		x_cropped.append(x_.cpu())
		y_cropped.append(y[indices])

	# Make dataloader out of this filtered data
	x_cropped = ch.cat(x_cropped, 0)
	y_cropped = ch.from_numpy(np.concatenate(y_cropped, 0))
	td        = TensorDataset(x_cropped, y_cropped)

	target_prop = attrs.index("Smiling")
	all_x, all_y = [], []

	for index, UPFOLDER in enumerate(folder_paths):
		model_latents = []
		model_stats = []

		for FOLDER in tqdm(os.listdir(UPFOLDER)):
			wanted_model = [x for x in os.listdir(os.path.join(UPFOLDER, FOLDER))if x.startswith("10_")][0]
			MODELPATH    = os.path.join(UPFOLDER, FOLDER, wanted_model)

			cropped_dataloader = DataLoader(td, batch_size=512, shuffle=False)

			# Get latent representations
			latent, all_stats = get_features_for_model(cropped_dataloader, MODELPATH)
			model_stats.append(all_stats)
			model_latents.append(latent)

			all_y.append(np.ones((latent.shape[0])) * index)

		model_latents = np.array(model_latents)
		model_stats   = np.array(model_stats)

		all_x.append(model_latents)

	all_x = np.concatenate(np.concatenate(np.array(all_x), 0), 0)
	all_y = np.concatenate(all_y, 0)

	print(all_x.shape)
	print(all_y.shape)

	x_tr, x_te, y_tr, y_te = train_test_split(all_x, all_y, test_size=0.4)
	clf = MLPClassifier(hidden_layer_sizes=(300, 200, 100, 50))
	clf.fit(x_tr, y_tr)
	print(clf.score(x_tr, y_tr))
	print(clf.score(x_te, y_te))

	# Test out on unseen models
	all_scores = []
	for path in blind_test_models:
		cropped_dataloader = DataLoader(td, batch_size=512, shuffle=False)
		latent, _ = get_features_for_model(cropped_dataloader, path)

		preds = clf.predict_proba(latent)[:, 0]
		all_scores.append(preds)
		print(np.mean(preds))

	labels = ['C0' ,'C0' ,'C0', 'C1' ,'C1' ,'C1']
	for i, x in enumerate(all_scores):
		plt.hist(x, 100, label=labels[i], density=True)
	
	plt.legend()
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

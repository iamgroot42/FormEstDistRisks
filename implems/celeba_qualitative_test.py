import numpy as np
import utils
import torch as ch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os

from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_stats(mainmodel, dataloader, target_prop, return_preds=False):
	stats = []
	all_stats = []
	all_preds = [] if return_preds else None
	for (x, y) in dataloader:

		y_ = y[:, target_prop].cuda()

		preds = mainmodel(x.cuda()).detach()[:,0]
		incorrect = ((preds >= 0) != y_)
		stats.append(y[incorrect].cpu().numpy())
		if return_preds:
			all_preds.append(preds.cpu().numpy())
			all_stats.append(y.cpu().numpy())

	return np.concatenate(stats), np.concatenate(all_preds), np.concatenate(all_stats)


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

	# paths = ["/u/as9rw/work/fnb/implems/celeba_models/smile_old_vggface_cropped_augs/10_0.9244576840818821",
	# 		"/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/10_0.9289294306335204",
	# 		"/u/as9rw/work/fnb/implems/celeba_models/smile_attractive_vggface_cropped_augs/20_0.9265191091558977",
	# 		"/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs/7_0.9226591187270502"]

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
	all_cfms = []

	for UPFOLDER in folder_paths:
		model_preds = []
		model_stats = []

		for FOLDER in tqdm(os.listdir(UPFOLDER)):
			wanted_model = [x for x in os.listdir(os.path.join(UPFOLDER, FOLDER))if x.startswith("10_")][0]
			MODELPATH    = os.path.join(UPFOLDER, FOLDER, wanted_model)
	
			# Load model
			model = utils.FaceModel(512, train_feat=True).cuda()
			model = nn.DataParallel(model)
			model.load_state_dict(ch.load(MODELPATH))
			model.eval()

			cropped_dataloader = DataLoader(td, batch_size=512, shuffle=False)

			stats, preds, all_stats = get_stats(model, cropped_dataloader, target_prop, return_preds=True)
			model_stats.append(all_stats)
			model_preds.append(preds)

		model_preds = np.array(model_preds)
		model_stats = np.array(model_stats)

		yeslabel = np.nonzero(all_stats[:, target_prop] == 1)[0] 
		nolabel  = np.nonzero(all_stats[:, target_prop] == 0)[0]

		# Look at loss
		lossfn = nn.BCEWithLogitsLoss(reduction='none')
		cfms = []

		# Pick relevant samples
		label_attr     = attrs.index(inspect_these[1])
		label_prop     = np.nonzero(all_stats[yeslabel, label_attr] == 1)[0]
		label_noprop   = np.nonzero(all_stats[yeslabel, label_attr] == 0)[0]
		nolabel_prop   = np.nonzero(all_stats[nolabel, label_attr] == 1)[0]
		nolabel_noprop = np.nonzero(all_stats[nolabel, label_attr] == 0)[0]

		for i in range(len(model_preds)):
		
			label_prop_losses     = lossfn(ch.from_numpy(model_preds[i][yeslabel][label_prop]), ch.from_numpy(1. * all_stats[yeslabel, target_prop][label_prop]))
			label_noprop_losses   = lossfn(ch.from_numpy(model_preds[i][yeslabel][label_noprop]), ch.from_numpy(1. * all_stats[yeslabel, target_prop][label_noprop]))

			nolabel_prop_losses   = lossfn(ch.from_numpy(model_preds[i][nolabel][nolabel_prop]), ch.from_numpy(1. * all_stats[nolabel, target_prop][nolabel_prop]))
			nolabel_noprop_losses = lossfn(ch.from_numpy(model_preds[i][nolabel][nolabel_noprop]), ch.from_numpy(1. * all_stats[nolabel, target_prop][nolabel_noprop]))

			cfms.append([[ch.mean(label_prop_losses), ch.mean(label_noprop_losses)], [ch.mean(nolabel_prop_losses), ch.mean(nolabel_noprop_losses)]])

		all_cfms.append(cfms)

	all_cfms = np.array(all_cfms)

	# Get property-wise losses per model
	prop_losses   = (all_cfms[:,:,0,0] * len(label_prop) + all_cfms[:,:,1,0] * len(label_noprop)) / (len(label_prop) + len(label_noprop))
	noprop_losses = (all_cfms[:,:,0,1] * len(nolabel_prop) + all_cfms[:,:,1,1] * len(nolabel_noprop)) / (len(nolabel_prop) + len(nolabel_noprop))

	def get_coeffs(X, Y):
		slope     = Y[0] / X[0]
		intercept = (Y[1] - X[1]) / slope
		return (intercept, slope)

	colors = [['C0', 'C1'], ['C1', 'C2']]
	patches = []
	patches.append(mpatches.Patch(color='C0', label='All v/s All'))
	patches.append( mpatches.Patch(color='C1', label='All v/s Male'))
	patches.append(mpatches.Patch(color='C2', label='Male v/s Male'))

	# Scatter ,pde
	# for i in range(all_cfms.shape[0]):
	# 	scats_x, scats_y = [], []
	# 	for j in range(all_cfms.shape[1]):
	# 		scats_x.append(prop_losses[i][j])
	# 		scats_y.append(noprop_losses[i][j])
	# 	plt.scatter(scats_x, scats_y, c=colors[i][0])

	# Line plot mode
	# Per model type
	for i in range(all_cfms.shape[0]):
		# Per model
		for j in range(all_cfms.shape[1]):
			# Per model type
			for k in range(i, all_cfms.shape[0]):
				# Per model
				for l in range(all_cfms.shape[1]):

					# Skip same model type, same model
					if i == k and j == l: continue

					# print(i, j, k, l)

					m, c = get_coeffs((prop_losses[i][j], noprop_losses[i][j]), (prop_losses[k][l], noprop_losses[k][l]))

					plot_x = np.linspace(0, 1, 50)
					plot_y = m * plot_x + c

					plt.plot(plot_x, plot_y, c=colors[i][k])

	plt.legend(handles=patches)
	plt.savefig('../visualize/celeba_lines.png')

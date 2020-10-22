import numpy as np
import utils
import torch as ch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import os

from facenet_pytorch import InceptionResnetV1, MTCNN


def get_stats(mainmodel, cropmodel, dataloader, target_prop, return_preds=False):
	stats = []
	all_stats = []
	all_preds = [] if return_preds else None
	for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):

		x_cropped   = []
		valid_faces = []
		for j, x_ in enumerate(x):
			x_ = (x_ * 0.5) + 0.5
			image = Image.fromarray((255 * np.transpose(x_.numpy(), (1, 2, 0))).astype('uint8'))
			try:
				x_cr = cropmodel(image)
				x_cropped.append(x_cr)
				valid_faces.append(j)
			except:
				continue

		x_cropped = ch.stack(x_cropped, 0)

		y  = y[valid_faces]
		y_ = y[:, target_prop].cuda()

		preds = model(x_cropped.cuda()).detach()[:,0]
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

	target_prop = attrs.index("Smiling")

	paths = ["/u/as9rw/work/fnb/implems/celeba_models/smile_old_vggface_cropped_augs/10_0.9244576840818821",
			"/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/10_0.9289294306335204",
			"/u/as9rw/work/fnb/implems/celeba_models/smile_attractive_vggface_cropped_augs/20_0.9265191091558977",
			"/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs/7_0.9226591187270502"]

	cropmodel = MTCNN(device='cuda')
	model_preds = []
	model_stats = []
	for MODELPATH in paths:
		_, dataloader = ds.make_loaders(batch_size=512, workers=8, shuffle_train=False, shuffle_val=False, only_val=True)

		model = utils.FaceModel(512, train_feat=True).cuda()
		model = nn.DataParallel(model)
		model.load_state_dict(ch.load(MODELPATH))
		model.eval()

		stats, preds, all_stats = get_stats(model, cropmodel, dataloader, target_prop, return_preds=True)
		model_stats.append(all_stats)
		model_preds.append(preds)

		# for it in inspect_these:
		# 	prop = attrs.index(it)
		# 	print(it, ":", np.mean(stats[:, prop]))

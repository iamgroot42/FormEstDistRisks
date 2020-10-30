from cleverhans.future.torch.attacks import projected_gradient_descent
from PIL import Image
from facenet_pytorch import MTCNN
import torch as ch
import numpy as np
import torch.nn as nn
import utils


if __name__ == "__main__":
	eps = 4#30.0
	nb_iter = 200
	eps_iter = 2.5 * eps / nb_iter
	norm = 2

	cropmodel = MTCNN(device='cuda')
	paths = ["/u/as9rw/work/fnb/implems/celeba_models/smile_old_vggface_cropped_augs/10_0.9244576840818821",
			"/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/10_0.9289294306335204",
			"/u/as9rw/work/fnb/implems/celeba_models/smile_attractive_vggface_cropped_augs/20_0.9265191091558977",
			"/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs/7_0.9226591187270502"]
	
	models = []
	for MODELPATH in paths:
		model = utils.FaceModel(512, train_feat=True).cuda()
		model = nn.DataParallel(model)
		# MODELPATH = "/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs/10_0.9289294306335204"
		model.load_state_dict(ch.load(MODELPATH))
		model.eval()
		models.append(model)

	constants = utils.Celeb()
	ds = constants.get_dataset()
	_, dataloader = ds.make_loaders(batch_size=128, workers=8, shuffle_val=False, only_val=True)

	attrs = constants.attr_names
	target_prop = attrs.index("Smiling")

	def saveimg(x_, path):
		x_ = (x_ * 0.5) + 0.5
		image = Image.fromarray((255 * np.transpose(x_.numpy(), (1, 2, 0))).astype('uint8'))
		image.save(path)

	x_advs = []
	for x, y in dataloader:
		# Get cropped versions
		x_, indices = utils.get_cropped_faces(cropmodel, x)
		y_picked = y[indices, target_prop].cuda()

		for j, model in enumerate(models):
			model_fn = lambda x: model(x)[:,0]

			y_pseudo = 1. * (model(x)[:, 0] >= 0)

			x_adv = projected_gradient_descent(model_fn, x_, eps, eps_iter, nb_iter, norm,
							   clip_min=-1, clip_max=1, y=y_pseudo,
							   rand_init=True, sanity_checks=True,
							   loss_fn=nn.BCEWithLogitsLoss()).detach().cpu()
			x_advs.append(x_adv)
			# selected_index = 7 # 5
			# saveimg(x_[selected_index].cpu(), "../visualize/normal.png")
			# saveimg(x_adv[selected_index].cpu(), "../visualize/perturbed_" + str(j) + ".png")

			# print(model(x_)[selected_index])
			# print(model(x_adv)[selected_index])
	
		break

	# Look at inter-model transfer for adversarial examples
	for i in range(len(x_advs)):
		preds_og = models[i](x_advs[i])[:, 0]
		print("Original error: %.2f" % (1 - ch.mean(1. * (y_picked == (preds_og >=0)))))
		for j in range(i, len(x_advs)):
			preds_target = models[j](x_advs[i])[:, 0]
			print("Transfer rate: %.2f" % (1 - 1 *ch.mean(1. * (y_picked == (preds_target >=0)))))
		print()

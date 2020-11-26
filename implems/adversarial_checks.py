from cleverhans.future.torch.attacks import projected_gradient_descent
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
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
	paths = [
		"/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/20_0.9235165574046058.pth",
		"/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
		"/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/augment_none/20_0.9065300896286812.pth",
		"/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/20_0.9108834827144686.pth"
		]
	
	models = []
	for MODELPATH in paths:
		model = utils.FaceModel(512, train_feat=True, weight_init=None).cuda()
		model = nn.DataParallel(model)
		model.load_state_dict(ch.load(MODELPATH))
		model.eval()
		models.append(model)

	constants = utils.Celeb()
	ds = constants.get_dataset()
	_, dataloader = ds.make_loaders(batch_size=256, workers=8, shuffle_val=True, only_val=True)

	attrs = constants.attr_names
	target_prop = attrs.index("Smiling")
	# Look at examples that satisfy particular property
	inspect_these = ["Attractive", "Male", "Young"]

	def saveimg(x_, path):
		x_ = (x_ * 0.5) + 0.5
		image = Image.fromarray((255 * np.transpose(x_.numpy(), (1, 2, 0))).astype('uint8'))
		image.save(path)

	x_advs = []
	for x, y in dataloader:
		# Get cropped versions
		x_, indices = utils.get_cropped_faces(cropmodel, x)
		y_picked = y[indices, target_prop].cuda()

		# Pick only the ones that satisfy property
		y_anal   = y[indices]
		satisfy  = ch.nonzero(y_anal[:, attrs.index(inspect_these[1])])[:, 0]
		x_       = x_[satisfy]
		y_picked = y_picked[satisfy]

		for j, model in tqdm(enumerate(models)):
			model_fn = lambda z: model(z)[:,0]

			y_pseudo = 1. * (model(x_)[:, 0] >= 0)

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
	names = ["al", "all", "male", "male"]
	for i in range(len(x_advs)):
		preds_og = models[i](x_advs[i])[:, 0]
		# print("Original error on %s : %.2f" % (names[i], 1 - ch.mean(1. * (y_picked == (preds_og >=0)))))
		for j in range(i, len(x_advs)):
			preds_target = models[j](x_advs[i])[:, 0]
			print("Transfer rate to %s : %.2f" % (names[j], 1 - 1 *ch.mean(1. * (y_picked == (preds_target >=0)))))
		print()

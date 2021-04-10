import torch as ch
import utils
import numpy as np
from tqdm import tqdm
from cleverhans.future.torch.attacks import carlini_wagner_l2, projected_gradient_descent


def get_perturbations(model, im, label):
	with ch.no_grad():
		# Get latent activations (input for C&W attack)
		intermediate_outputs, _ = model.model(im, this_layer_output=model.layer_index - 1, just_latent=True)
	upper_clip = 5e0
	n_iters    = 200
	adv_x = carlini_wagner_l2(model, intermediate_outputs, 10,
                      confidence=0, clip_max=upper_clip, binary_search_steps=5,
                      max_iterations=1000)
	# Check how much of it actually works
	adv_preds = ch.argmax(model(adv_x), 1)

	l2_norms  = ch.sqrt(ch.sum(ch.pow(adv_x - intermediate_outputs, 2), 1))
	# print(ch.min(l2_norms), ch.max(l2_norms))
	model_preds = ch.argmax(model(intermediate_outputs), 1)
	asr = ch.sum(adv_preds != model_preds).float()
	return adv_x.cpu().numpy(), asr


def get_sensitivities(wrapped_model, data_loader):
	iterator = tqdm(data_loader)
	iterator.set_description("Delta Success Rate : 0.00")
	total, gotcha = 0, 0
	perturbation_vectors = []
	for (im, label) in iterator:
		im, label = im.cuda(), label.cuda()
		latent_perturbations, asr = get_perturbations(wrapped_model, im, label)
		total  += im.shape[0]
		gotcha += asr
		perturbation_vectors.append(latent_perturbations)
		iterator.set_description("Misclassification Rates for delta values : %.2f" % (100 * gotcha/ total))
	stacked_perts = np.concatenate(perturbation_vectors, 0)
	return stacked_perts


if __name__ == "__main__":
	import sys, os
	filename        = sys.argv[1]
	model_type      = sys.argv[2]
	dataset         = "cifar" #sys.argv[3]
	injection_layer = int(sys.argv[3])

	if dataset == "cifar":
		dx = utils.CIFAR10()
	elif dataset == "imagenet":
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds    = dx.get_dataset()
	model = dx.get_model(model_type, "vgg19")

	batch_size = 5000
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	# VGG-19 specific parameters
	# Layer 48, 45, 42:  512 * 2 * 2
	# Layer 38, 35, 32, 29 : 512 * 4 * 4 
	# Layer 25, 22, 19, 16: 256 * 8 * 8
	# Layer 12, 9: 128 * 16 * 16
	# Layer 5, 2: 64 * 32 * 32
	wrapped_model = utils.SpecificLayerModel(model, injection_layer + 1)
	sensitivities = get_sensitivities(wrapped_model, test_loader)

	np.save(os.path.join(filename, str(injection_layer)), sensitivities)

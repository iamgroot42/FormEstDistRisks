import torch as ch
from robustness.datasets import CIFAR

ds = CIFAR()

from robustness.model_utils import make_and_restore_model
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path="./models/cifar_nat.pt")
model.eval()

_, test_loader = ds.make_loaders(workers=10, batch_size=8)
_, (im, label) = next(enumerate(test_loader))

kwargs = {
    'constraint':'1',
    'eps': 10,
    'step_size': 1.0,
    'iterations': 20,
    'do_tqdm': True,
}

_, im_adv = model(im, label, make_adv=True, **kwargs)


from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT

# Get predicted labels for adversarial examples
pred, _ = model(im_adv)
label_pred = ch.argmax(pred, dim=1)

# Visualize test set images, along with corresponding adversarial examples
show_image_row([im.cpu(), im_adv.cpu()],
         tlist=[[CLASS_DICT['CIFAR'][int(t)] for t in l] for l in [label, label_pred]],
         fontsize=18,
         filename='./adversarial_example_CIFAR.png')

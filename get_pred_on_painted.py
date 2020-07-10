import utils
import numpy as np
from PIL import Image
import torch as ch


constants = utils.CIFAR10()
ds = constants.get_dataset()
model = constants.get_model("linf" , "vgg19")

im = Image.open("./visualize/paint_this_edited.jpg")
nim = (np.array(im).astype(np.float32) / 255).transpose(2, 0, 1)
nim_ch = ch.from_numpy(np.expand_dims(nim, 0)).cuda()

mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print(nim_ch.shape)

logits, _ = model(nim_ch)
probs = ch.nn.Softmax(dim=1)(logits)

for i, x in enumerate(probs[0]):
	print(mappinf[i], ": %.4f" % x.item())

import sys
import numpy as np
from robustness.datasets import GenericBinary
from robustness.tools.vis_tools import show_image_row
import torch as ch


filename = "./sorted_indices.txt"

with open(filename, 'r') as f:
	for line in f:
		sorted_text = [int(x) for x in line.rstrip('\n').split(',')]
		sorted_first = np.array(sorted_text)
		break

n = 6
first_n = sorted_first[:n*2]
last_n  = sorted_first[-n*2:]
middle_ones = sorted_first[int(len(sorted_first)/2) - n : int(len(sorted_first)/2) + n]



batch_size = 512
ds_path    = "./datasets/cifar_binary/animal_vehicle/"
ds = GenericBinary(ds_path)
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)
images = []
for (image, _) in test_loader:
	images.append(image)
images = ch.cat(images)

first_images = images[first_n]
last_images = images[last_n]
middle_images = images[middle_ones]

show_image_row([first_images.cpu(), last_images.cpu(), middle_images.cpu()], 
				["First N", "Last N", "Middle N"], 
				fontsize=22,
				filename="tail_images.png")

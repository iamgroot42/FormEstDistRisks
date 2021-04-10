from inception_score import inception_score
import numpy as np
import torch as ch


if __name__ == "__main__":
	# Load image
	sc_img = np.load("./visualize/closest_to_this.npy")
	sc_img = ((255 * sc_img).astype('uint8') / 255).astype('float32')
	# Normalize to [-1,1]
	sc_img = (sc_img - 0.5) * 2

	sc_img = np.expand_dims(sc_img.transpose(2, 0, 1), 0)
	sc_img = ch.from_numpy(sc_img)

	score = inception_score(sc_img, cuda=True, batch_size=1, resize=True, splits=1)
	print("Inception Score: %f" % score)

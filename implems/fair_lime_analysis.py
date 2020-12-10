import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch as ch
from torchvision import models, transforms
from torch.autograd import Variable

from lime import lime_image

import utils
import implem_utils

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_model(path):
    model = utils.FaceModel(512,
                            train_feat=True,
                            weight_init=None,
                            hidden=[64, 16]).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(path), strict=False)
    model.eval()
    return model


def raw_255_image(z):
    z_ = z.numpy().transpose(1, 2, 0)
    z_ = (z_ + 1) / 2
    return (z_ * 255).astype(np.uint8)


def batch_predict(model, images):
    batch = ch.stack(tuple(transform(i) for i in images), dim=0)
    logits = model(batch.cuda()).detach()
    probs = ch.sigmoid(logits)
    probs = ch.stack([1 - probs[:, 0], probs[:, 0]], 1)
    return probs.cpu().numpy()


def get_scores(model):
    explainer = lime_image.LimeImageExplainer()

    scores = []
    labels = []

    def model_batch_predict(x):
        return batch_predict(model, x)

    for i, (x, y) in tqdm(enumerate(cropped_dataloader), total=len(cropped_dataloader)):
        x_raw = [raw_255_image(x_) for x_ in x]
        labels.append(y.numpy())

        for img_t in x_raw:
            explanation = explainer.explain_instance(img_t,
                                                     model_batch_predict,
                                                     top_labels=1,
                                                     hide_color=0,
                                                     num_samples=200,
                                                     progress_bar=False)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=True,
                                                        num_features=5,
                                                        hide_rest=False)

            temp_pos = temp * (np.expand_dims(mask, 2))
            temp_neg = temp * (1 - np.expand_dims(mask, 2))
            test_pred = model_batch_predict([temp_pos.astype(np.uint8),
                                            temp_neg.astype(np.uint8)])
            scores.append(test_pred[:, 1])

        if i == 10:
            break

    return np.stack(scores, 0), np.concatenate(labels, 0)


if __name__ == "__main__":
    import sys
    MODELPATH = sys.argv[1]

    constants = utils.Celeb()
    ds = constants.get_dataset()
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))])

    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    attrs = constants.attr_names

    cropped_dataloader = DataLoader(td,
                                    batch_size=5,
                                    shuffle=False)

    def get_input_tensors(img):
        # unsqeeze converts single image to batch of 1
        return transform(img).unsqueeze(0)

    model = get_model(MODELPATH)

    scores, labels = get_scores(model)

    where_prop = np.nonzero(labels[:, attrs.index("Male")])[0]
    where_noprop = np.nonzero(1 - labels[:, attrs.index("Male")])[0]
    prop_order = np.argsort(scores[where_prop, 0] - scores[where_prop, 1])
    noprop_order = np.argsort(scores[where_noprop, 0] - scores[where_noprop, 1])

    plt.plot(np.arange(where_prop.shape[0]),
             scores[where_prop[prop_order], 0] - scores[where_prop[prop_order], 1],
             label='prop=True')
    plt.plot(np.arange(where_noprop.shape[0]),
             scores[where_noprop[noprop_order], 0] - scores[where_noprop[noprop_order], 1],
             label='prop=False')

    plt.legend()
    plt.savefig("../visualize/lime_score_distrs.png")

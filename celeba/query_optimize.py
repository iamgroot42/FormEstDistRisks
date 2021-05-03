# from cleverhans.future.torch.attacks import projected_gradient_descent
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch as ch
import numpy as np
from PIL import Image
import torch.nn as nn
import utils


def static_identifier(dataloader, model1, model2, agree=True):
    base = 0
    indices, scores = [], []
    for x, y in tqdm(dataloader):
        preds1 = model1(x.cuda()).detach()[:, 0]
        preds2 = model2(x.cuda()).detach()[:, 0]
        # Pick the ones where predictions differ
        # Also keep track of score differences for later use
        if agree:
            differ = ch.nonzero((preds1 >= 0) == (preds2 >= 0))[:, 0].cpu()
        else:
            differ = ch.nonzero((preds1 >= 0) != (preds2 >= 0))[:, 0].cpu()
        indices.append(base + differ)
        scores.append(ch.abs(preds1[differ] - preds2[differ]).cpu())
        base += x.shape[0]
    indices = ch.cat(indices, 0).numpy()
    scores = ch.cat(scores, 0).numpy()
    return (indices, scores)


if __name__ == "__main__":
    # Optimize to find query points for which predictions on models trainined on different ratio properties differ the most
    # Or better yet, identify such examples from a test set
    eps = 4
    nb_iter = 200
    eps_iter = 2.5 * eps / nb_iter
    norm = 2

    paths = [
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/20_0.9235165574046058.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/augment_none/20_0.9065300896286812.pth",
        "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/20_0.9108834827144686.pth"
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/augment_vggface/20_0.9307090239410681.pth",
        # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/20_0.9212707182320442.pth"
    ]

    # Use existing dataset instead
    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    models = []
    for MODELPATH in paths:
        model = utils.FaceModel(512, train_feat=True, weight_init=None).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(ch.load(MODELPATH), strict=False)
        model.eval()
        models.append(model)

    attrs = constants.attr_names
    target_prop = attrs.index("Smiling")

    batch_size = 128
    cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)
    x_1, scores_1 = static_identifier(cropped_dataloader,
                                      models[0], models[1], agree=True)
    cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)
    x_2, scores_2 = static_identifier(cropped_dataloader,
                                      models[2], models[3], agree=True)
    
    cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)
    x_3, scores_3 = static_identifier(cropped_dataloader,
                                      models[1], models[2], agree=False)

    print(x_1.shape, x_2.shape, x_3.shape)
    # Pick intersection of all three cases
    overlap = np.intersect1d(np.intersect1d(x_1, x_2), x_3)

    base = 0
    cropped_dataloader = DataLoader(td, batch_size=batch_size, shuffle=False)
    X, Y = utils.load_all_loader_data(cropped_dataloader)
    X_want = X[overlap].numpy()
    # Print out prediction scores for all models
    for model in models:
        scores = ch.sigmoid(model(ch.from_numpy(X_want).cuda()).detach().cpu())
        print([x.item() for x in scores])

    # Save them in dump folder
    for i, x in enumerate(X_want):
        image = Image.fromarray(
                    (255 * np.transpose(x, (1, 2, 0))).astype('uint8'))
        image.save("/u/as9rw/work/fnb/visualize/dump/%d.png" % i)

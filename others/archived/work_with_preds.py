import numpy as np
import torch as ch
import utils
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


x = np.load("laterz/preds.npy")
y = np.load("laterz/labels.npy")


constants = utils.Celeb()
ds = constants.get_dataset()

attrs = constants.attr_names
inspect_these = ["Attractive", "Male", "Young"]

attractive_people = np.nonzero(y[:, attrs.index("Attractive")] == 1)[0]
male_people = np.nonzero(y[:, attrs.index("Male")] == 1)[0]
old_people = np.nonzero(y[:, attrs.index("Young")] == 0)[0]

target_prop = attrs.index("Smiling")


def get_cropped_faces(cropmodel, x):
    renormalize = lambda z: (z * 0.5) + 0.5
    images = [Image.fromarray((255 * np.transpose(renormalize(x_.numpy()), (1, 2, 0))).astype('uint8')) for x_ in x]
    crops = cropmodel(images)

    x_cropped = []
    indices = []
    for j, cr in enumerate(crops):
        if cr is not None:
            x_cropped.append(cr)
            indices.append(j)

    return ch.stack(x_cropped, 0), indices

# cropmodel = MTCNN(device='cuda')
# _, dataloader = ds.make_loaders(batch_size=2500, workers=8, shuffle_val=False, only_val=True)
# x_cropped = []
# y_cropped     = []
# for (x, y) in tqdm(dataloader, total=len(dataloader)):
# 	x, indices = get_cropped_faces(cropmodel, x)
# 	y_cropped.append(y[indices])
# 	x_cropped.append(x)

# y_cropped = np.concatenate(y_cropped, 0)
# x_cropped = ch.cat(x_cropped, 0)

# my_dataloader = DataLoader(TensorDataset(x_cropped, ch.from_numpy(y_cropped)), batch_size=200)

# print(x_cropped.shape)
# print(y_cropped.shape)

folders = [
    "/u/as9rw/work/fnb/implems/celeba_models/smile_old_vggface_cropped_augs",
    "/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs",
    "/u/as9rw/work/fnb/implems/celeba_models/smile_attractive_vggface_cropped_augs",
    "/u/as9rw/work/fnb/implems/celeba_models/smile_male_vggface_cropped_augs"
]


# np.save("laterz/all_pred_labels", y_cropped)
# print("Saved labels for later!")

x = np.load("laterz/all_preds.npy")
y_cropped = np.load("laterz/all_pred_labels.npy")
positive = np.nonzero(y_cropped[:, target_prop] == 1)[0]

for i in range(10):
	plt.hist(x[0][i][positive], color='red', bins=100)

for i in range(10):
    plt.hist(x[1][i][positive], color='blue', bins=100)

for i in range(10):
    plt.hist(x[2][i][positive], color='green', bins=100)


for i in range(10):
    plt.hist(x[3][i][positive], color='orange', bins=100)

plt.savefig('../visualize/celeb0scores.png')

# all_the_scores = []
# for j, f in tqdm(enumerate(folders)):
# 	folderwise_scores = []
# 	for i, path in enumerate(os.listdir(f)):
# 		# Take 10 classifiers per folder
# 		if i == 10: break

# 		model = utils.FaceModel(512, train_feat=True).cuda()
# 		model = nn.DataParallel(model)
# 		model.load_state_dict(ch.load(os.path.join(f, path)))
# 		model.eval()

# 		scores_all = []
# 		for (x, _) in my_dataloader:
# 			m_scores = model(x.cuda())[:, 0].detach().cpu().numpy()
# 			scores_all.append(m_scores)
# 		scores_all = np.concatenate(scores_all, 0)
# 		folderwise_scores.append(scores_all)
# 	folderwise_scores = np.stack(folderwise_scores, 0)
# 	all_the_scores.append(folderwise_scores)
	
# all_the_scores = np.stack(all_the_scores, 0)
# print(all_the_scores.shape)

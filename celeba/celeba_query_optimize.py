import numpy as np
import utils
import torch as ch
import torch.nn as nn
import os
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def optimizationLoop(models, labels, numQuery=100, numSteps=500, seed=None,
                     learningRate=1e-1):
    loss_fn = nn.BCEWithLogitsLoss()
    # Uniform-random in [-0.5,0.5] range
    if seed is None:
        query = ch.rand(numQuery, 3, 160, 160).cuda()
        query = ((query - 0.5) / 0.5).requires_grad_(True)
    else:
        query = seed.clone().cuda().requires_grad_(True)

    # Define optimizer
    optimizer = ch.optim.SGD([query], lr=learningRate)

    y = ch.from_numpy(np.array([[l] * numQuery for l in labels]))
    y = ch.unsqueeze(ch.flatten(y), 1).float().cuda()

    iterator = tqdm(range(numSteps))
    for _ in iterator:
        # Zero out optimizer gradients
        optimizer.zero_grad()

        # Compute values
        predictions = []
        predictions = ch.cat([m(query) for m in models], 0)

        # Compute loss
        loss = loss_fn(predictions, y)

        # Gradient computation and calculation
        loss.backward()
        optimizer.step()

        # Clamp back to [0,1] range
        query.data = ch.clamp(query, 0, 1)

        iterator.set_description("Loss: %.3f" % loss.item())

    # Return queries
    return query.detach()


if __name__ == "__main__":
    import argparse
    methods = ['standalone', 'meta']

    parser = argparse.ArgumentParser()
    parser.add_argument('--qp', type=int, default=75, help='number of query points to generate')
    parser.add_argument('--randomseed', type=bool, default=False, help='use random seed(True) or actual examples (True)')
    parser.add_argument('--dump', type=bool, default=False, help='save generated query points?')
    parser.add_argument('--method', type=str, default='standalone', help='which method to use (%s)' % "/".join(methods))
    args = parser.parse_args()
    utils.flash_utils(args)

    numQuery = args.qp
    try:
        method_type = methods.index(args.method)
    except ValueError:
        print("Method %s not implemented yet: Pick one of: %s" % (args.method, "/".join(methods)))
        exit(0)

    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    folder_paths = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/",
        ]
    ]

    blind_test_models = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/10_0.928498243559719.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/10_0.9093969555035128.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/20_0.9151053864168618.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/20_0.9108606557377049.pth"
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/10_0.9240681998413958.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/10_0.8992862807295797.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/20_0.9259516256938938.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/20_0.9042426645519429.pth"
        ]
    ]

    models = []

    for index, UPFOLDER in enumerate(folder_paths):
        model_latents = []
        model_stats = []

        for pf in UPFOLDER:
            for MODELPATHSUFFIX in tqdm(os.listdir(pf)):
                if not("10_" in MODELPATHSUFFIX or "20_" in MODELPATHSUFFIX):
                    continue

                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                model = utils.FaceModel(512,
                                        train_feat=True,
                                        weight_init=None,
                                        hidden=[64, 16]).cuda()
                model = nn.DataParallel(model)
                model.load_state_dict(ch.load(MODELPATH), strict=False)
                model.eval()
                models.append(model)

    # Get a sample of random data to seed query generation process
    seed = None
    if not args.randomseed:
        _, dataloader = ds.make_loaders(
                batch_size=numQuery, workers=8, shuffle_train=False, shuffle_val=True, only_val=True)
        seed, _ = next(iter(dataloader))

    queries = optimizationLoop(models,
                               [1] * 4 + [0] * 4,
                               numQuery=numQuery,
                               numSteps=200,
                               learningRate=1e2,
                               seed=seed)

    if args.dump:
        # Save them somewhere for visualization
        queries_cpu = queries.cpu().numpy()
        for i in range(queries_cpu.shape[0]):
            image = Image.fromarray(
                        (255 * np.transpose(queries_cpu[i], (1, 2, 0))).astype('uint8'))
            image.save("../visualize/query_views/" + str(i + 1) + ".png")

    model_scores = [m(queries)[:, 0].detach().cpu() for m in models] 
    if method_type == 0:
        for sc in model_scores:
            print(ch.mean(ch.sigmoid(sc)))

    elif method_type == 1:
        clfs = []
        model_scores = np.array([x.numpy() for x in model_scores])
        idxs = [np.random.permutation(model_scores.shape[1])[:50] for i in range(10)]
        num_each = model_scores.shape[0] // 2

        for i in range(10):
            x_tr, x_te, y_tr, y_te = train_test_split(model_scores[:, idxs[i]],
                                                      [0] * num_each + [1] * num_each,
                                                      test_size=0.25)

            clf = MLPClassifier(hidden_layer_sizes=(32, 16))
            clf.fit(x_tr, y_tr)
            print("%.2f train, %.2f test" % (clf.score(x_tr, y_tr), clf.score(x_te, y_te)))
            clfs.append(clf)

    # Test out on unseen models
    all_scores = []
    for pc in blind_test_models:
        ac = []
        for path in pc:

            model = utils.FaceModel(512,
                                    train_feat=True,
                                    weight_init=None,
                                    hidden=[64, 16]).cuda()
            model = nn.DataParallel(model)
            model.load_state_dict(ch.load(path), strict=False)
            model.eval()
            preds = model(queries)[:, 0].detach().cpu()

            if method_type == 0:
                preds = ch.sigmoid(preds).numpy()
                print(np.mean(preds))

            elif method_type == 1:
                preds = [clf.predict_proba(np.expand_dims(preds[idx], 0))[0, 1] for idx, clf in zip(idxs, clfs)]
                print(np.mean(preds), np.std(preds))

            ac.append(preds)

        all_scores.append(ac)

    labels = ['Trained on $D_0$', 'Trained on $D_1$']
    for i, ac in enumerate(all_scores):
        # Brign down everything to fractions
        for x in ac:
            plot_x = x
            plt.hist(plot_x, 100, label=labels[i])

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    plt.legend()
    plt.title("Prediction scores on tuned query poits")
    plt.xlabel("$Pr$[trained on $D_1$]")
    plt.ylabel("Number of query points")
    plt.savefig("../visualize/querypoint_distrs_celeba.png")

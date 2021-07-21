import pandas as pd
import torchvision.models as models
import torch as ch
from data_utils import ChestWrapper
import torch.nn as nn
import utils


class ChestModel(nn.Module):
    def __init__(self):
        super(ChestModel, self).__init__()

        # self.fe = models.mobilenet_v2(pretrained=True)
        # self.fe = models.vgg19_bn(pretrained=True)
        # self.fe = models.densenet161(pretrained=True)
        self.fe = models.inception_v3(pretrained=True, aux_logits=False)

        # self.fe.classifier = nn.Identity()
        self.fe.fc = nn.Identity()

        # Make sure FE is not trainable
        for param in self.fe.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

        # self.classifier = nn.Sequential(
        #     # nn.Dropout(p=0.2, inplace=False),
        #     # nn.Linear(2208, 1),
        #     nn.Linear(2048, 1),
        # )

    def forward(self, x):
        # return self.fe(x)
        z = self.fe(x)
        return self.classifier(z)


if __name__ == "__main__":
    # Define model
    model = ChestModel()
    model.cuda()
    model = nn.DataParallel(model)

    # Get data
    split = 2
    df_train = pd.read_csv("./data/splits/%d/train.csv" % split)
    df_val = pd.read_csv("./data/splits/%d/val.csv" % split)

    def filter(x): return x["gender"] == 1

    n_train, n_test = 30000, 20000
    target_ratio = 0.1
    df_train_processed = utils.heuristic(
        df_train, filter, target_ratio,
        n_train, class_imbalance=1.0, n_tries=300)

    df_val_processed = utils.heuristic(
        df_val, filter, target_ratio,
        n_test, class_imbalance=1.0, n_tries=300)

    ds = ChestWrapper(df_train_processed, df_val_processed)

    # Get loaders
    batch_size = 32 * 3 * 8
    train_loader, val_loader = ds.get_loaders(batch_size, shuffle=False)

    # Train model
    vloss, vacc = utils.train(model, (train_loader, val_loader),
                              lr=2e-3, epoch_num=20,
                              weight_decay=0.02,
                              verbose=True)

from robustness import model_utils, datasets, train, defaults
from robustness.datasets import RobustCIFAR, CIFAR

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

# Hard-coded dataset, architecture, batch size, workers
ds = RobustCIFAR("./datasets/cifar_dr_l1")

m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

# Create a cox store for logging
out_store = cox.store.Store("./new_models/train_on_l1_dr")

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': 0
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args, defaults.TRAINING_ARGS, RobustCIFAR)

# Train a model
train.train_model(train_args, m, (train_loader, val_loader), store=out_store)

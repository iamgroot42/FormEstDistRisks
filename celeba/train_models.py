from model_utils import create_model, save_model
from data_utils import SUPPORTED_PROPERTIES, CelebaWrapper
from utils import flash_utils, train


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True,
                        help='filename (prefix) to save model')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train model for')
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--ratio', type=float, required=True,
                        help='desired ratio for attribute')
    parser.add_argument('--split', choices=['victim', 'adv'], required=True)
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weightinit', type=str, default='vggface2',
                        help='which weight initialization to use: vggface2, " \
                            "casia-webface, or none')
    parser.add_argument('--augment', action="store_true",
                        help='use data augmentations when training models?')
    parser.add_argument('--parallel', action="store_true",
                        help='use multiple GPUs to train?')
    args = parser.parse_args()
    flash_utils(args)

    # CelebA dataset
    ds = CelebaWrapper(args.filter, args.ratio,
                       args.split, augment=args.augment)

    # Get loaders
    train_loader, test_loader = ds.get_loaders(args.bs)

    # Create model
    model = create_model(parallel=args.parallel)

    # Train model
    vloss, vacc = train(model, (train_loader, test_loader),
                        lr=args.lr, epoch_num=args.epochs,
                        weight_decay=0.01, verbose=True)

    # Save model
    save_name = args.name + "_" + str(vacc) + "_" + str(vloss) + ".pth"
    save_model(model, args.split, args.property, str(args.ratio), save_name)

from data_utils import BotNetWrapper
from model_utils import GCN, train_model, save_model
import argparse


def main(args):
    # Get data
    ds = BotNetWrapper(split=args.split, feat_len=args.n_feat,
                       subsample=args.subsample)

    # Define model
    model = GCN(n_inp=args.n_feat, n_hidden=args.hidden_channels,
                n_layers=args.num_layers, dropout=args.dropout,
                residual=True)
    if args.gpu:
        model.cuda()

    _, loader = ds.get_loaders(batch_size=1)

    # Train model
    train_model(model, ds, args)

    save_model(model, args.split, args.savename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--split', required=True, choices=['victim', 'adv'])
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_feat', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpu', action="store_true", help="Use CUDA?")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--verbose', action="store_true",
                        help="Print out batch-sise metrics")
    parser.add_argument("--savename", help="name to save trained model with")
    args = parser.parse_args()
    print(args)

    main(args)

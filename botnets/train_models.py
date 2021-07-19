from data_utils import BotNetWrapper
from model_utils import get_model, train_model, save_model
import argparse


def main(args):
    # Get data
    ds = BotNetWrapper(split=args.split, prop_val=args.prop_val)

    # Define model
    model = get_model(args)

    # Train model
    model, (tr_f1, te_f1) = train_model(model, ds, args)

    # Save model
    name_for_file = "_".join([args.savename, str(tr_f1), str(te_f1)]) + ".pt"
    save_model(model, args.split, args.prop_val, name_for_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Botnets-dataset (GCN)')
    parser.add_argument('--split', required=True, choices=['victim', 'adv'])
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_feat', type=int, default=1)
    parser.add_argument('--prop_val', type=int,
                        choices=[0, 1], required=True,
                        help="0 -> coeff <= 0.0066; 1 -> coeff > 0.0071")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpu', action="store_true", help="Use CUDA?")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--verbose', action="store_true",
                        help="Print out batch-sise metrics")
    parser.add_argument('--best_model', action="store_true",
                        help="Pick best model based on training F-1 score")
    parser.add_argument("--savename", help="name to save trained model with")
    args = parser.parse_args()
    print(args)

    main(args)

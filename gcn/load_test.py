from ogb.nodeproppred import Evaluator
import data_utils
import torch as ch
import argparse
import model_utils


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--split', choices=['victim', 'adv'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument("--load_path", help="path to save trained model")
    args = parser.parse_args()
    print(args)

    # Get dataset ready
    ds = data_utils.ArxivNodeDataset(args.split)

    n_edges = ds.g.number_of_edges()
    print("""----Data statistics------'
          #Edges %d
          #Classes %d
          """ %
          (n_edges, ds.num_classes,))

    # Define model
    model = model_utils.get_model(ds, args)
    evaluator = Evaluator(name='ogbn-arxiv')

    # Load weights into model
    model.load_state_dict(ch.load(args.load_path))

    # Compute accuracy
    train_idx, test_idx = ds.get_idx_split()
    _, test_acc = model_utils.test(model, ds, train_idx, test_idx, evaluator)

    print("Test accuracy for this model: %.2f" % test_acc)


if __name__ == "__main__":
    main()

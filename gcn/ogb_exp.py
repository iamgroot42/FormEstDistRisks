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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--degree', type=int, default=None)
    parser.add_argument("--savepath", help="path to save trained model")
    args = parser.parse_args()
    print(args)

    ds = data_utils.ArxivNodeDataset(args.split)

    n_edges = ds.g.number_of_edges()
    n_nodes = ds.g.number_of_nodes()
    print("""----Data statistics------'
          #Edges %d
          Nodes %d
          #Classes %d
          """ %
          (n_edges, n_nodes, ds.num_classes,))

    if args.degree is not None:
        print("-> Before modification")
        print("Nodes: %d, Average degree: %.2f" %
              (ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
        print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))

    # Modify dataset
    ds.change_mean_degree(args.degree)

    pick_tr, pick_te = 30000, 10000
    if args.split == "victim":
        pick_tr, pick_te = 62000, 35000

    # Subsample from all available labels to introduce some randomness
    ds.label_ratio_preserving_pick(pick_tr, pick_te)

    if args.degree is not None:
        print("-> After modification")
        print("Nodes: %d, Average degree: %.2f" %
              (ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
        print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))

    # Define model
    model = model_utils.get_model(ds, args)
    evaluator = Evaluator(name='ogbn-arxiv')

    # Train model
    run_accs = model_utils.train_model(ds, model, evaluator, args)
    acc_tr, acc_te = run_accs["train"][-1], run_accs["test"][-1]

    # Log performance on train, test nodes
    print()
    print("Train accuracy: %.2f" % (acc_tr))
    print("Test accuracy: %.2f" % (acc_te))

    # Save model
    ch.save(model.state_dict(), args.savepath +
            "_tr%.2f_te%.2f.pth" % (acc_tr, acc_te))


if __name__ == "__main__":
    # Play around with changing mean node degree for now
    # Start with base distribution, create two cases
    # One with original mean degree
    # One with increased mean degree (get rid of low-connectivity nodes)
    # Train models on these, use metaclassifiers to estimate
    # Inferabilitiy og this property
    main()

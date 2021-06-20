from ogb.nodeproppred import Evaluator
import data_utils
import argparse
import model_utils


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--split', required=True, choices=['victim', 'adv'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--degree', type=float, default=None)
    parser.add_argument(
        '--property', choices=data_utils.SUPPORTED_PROPERTIES, default="mean")
    parser.add_argument('--prune', type=float, default=0)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument("--savepath", help="path to save trained model")
    parser.add_argument("--prefix", help="prefix for saving models",
                        default=model_utils.BASE_MODELS_DIR)
    args = parser.parse_args()
    print(args)

    # Prune ratio should be valid and not too large
    assert args.prune >= 0 and args.prune <= 0.1

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

    # Modify dataset property
    if args.property == "mean":
        # Modify mean degree
        ds.change_mean_degree(args.degree, args.prune)
    else:
        # Get rid of nodes above a specified node-degree
        ds.keep_below_degree_threshold(args.degree, args.prune)

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
    model_utils.save_model(
        model, args.split,
        args.savepath + "_tr%.2f_te%.2f.pth" % (acc_tr, acc_te),
        prefix=args.prefix)


if __name__ == "__main__":
    # Play around with changing mean node degree for now
    # Start with base distribution, create two cases
    # One with original mean degree
    # One with increased mean degree (get rid of low-connectivity nodes)
    # Train models on these, use metaclassifiers to estimate
    # Inferabilitiy og this property
    main()

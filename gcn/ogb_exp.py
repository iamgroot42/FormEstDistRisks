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
    parser.add_argument('--degree', type=int, default=10)
    parser.add_argument("--save_path", help="path to save trained model")
    args = parser.parse_args()
    print(args)

    ds = data_utils.ArxivNodeDataset(args.split)

    n_edges = ds.g.number_of_edges()
    print("""----Data statistics------'
          #Edges %d
          #Classes %d
          """ %
          (n_edges, ds.num_classes,))

    print("Nodes in graph before modification: %d" % ds.num_nodes)

    # Modify dataset
    ds.change_mean_degree(args.degree)

    print("Nodes in graph after modification: %d" % ds.num_nodes)

    # Define model
    model = model_utils.get_model(ds, args)
    evaluator = Evaluator(name='ogbn-arxiv')

    # Train model
    run_accs = model_utils.train_model(ds, model, evaluator, args)
    acc_tr, acc_te = run_accs["train"][-1], run_accs["test"][-1]

    # Log performance on train, test nodes
    print("Train accuracy: %.2f" % (acc_tr))
    print("Test accuracy: %.2f" % (acc_te))

    # Save model
    ch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    # Play around with changing mean node degree for now
    # Start with base distribution, create two cases
    # One with original mean degree
    # One with increased mean degree (get rid of low-connectivity nodes)
    # Train models on these, use metaclassifiers to estimate
    # Inferabilitiy og this property
    main()

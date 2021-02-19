from hmmlearn.hmm import GMMHMM
from audio_utils import VoxForgeData


if __name__ == "__main__":
    # Load data
    d = VoxForgeData()
    d.load_data("./data_np_single.npy")
    x, y, z = d.flatten_data_all()

    print(len(x))

    # Define model
    model = GMMHMM(
        n_components=3,
        n_mix=25,
        verbose=True,
    )
    # X: n_samples, n_features,
    # lengths: n_sequences, lengths of sequences & sum should be n_samples
    # model.fit(X, lengths)
    # viterbi train
    # baum-welch finetune

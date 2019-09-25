# fnb

Code for "Adversarial Examples Are Not Bugs, They Are Features" in Tensorflow + Cleverans + Keras, along with some additional experiments (ongoing):

* Hide specific trends in a trained model (via some form of watermarking) M and measure how much of those patterns manifest in another model M' trained using Dr. If these features are highly non-robust (terminology used in the paper), they should not be transferred in any way (less likely to be, at least) in Dr, so M' should not be react to such watermarks
* Do datasets Dr created from models with specific attack robustness differ significantly? If so, does that mean that the "robust features" are really only features that are robust to that specific attack, and thus not "robust" in essence
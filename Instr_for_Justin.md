# Instructions for running experiments


1. Navigate to the folder with code

2. Create a new folder for the current experiment with `mkdir folder_name`

3. Start a new screen (so that you can later close your SSH session without terminating the current process) with `screen -S screen_name`

4. Pick a ratio of your choice `sample_ratio` (in [0,1], eg. 0.1). Note that running the same experiment (same ratio) will generate different outputs, since the images are randomly sampled. Keep this in mind.

5. Start the experiment with `python collect_maximization_images.py --seed_mode_normal True --sample_ratio sample_ratio --save_path folder_name`

6. Exit out the screen with `ctrl-A + ctrl-D`

7. You can now log out the machine and check later how much of the experiment is complete (in an hour or so: depends)



## For 9/7-9/13

- Try running the experiment with sample_ratio=0.05 2-3 times

## For 9/17

- I have created a folder in `/p/adversarialml/jyc9fyf`

- You can now run the experiments above, and save them in folders inside this directory (this way, I can access them as well)

- Run the following experiments (replace `<folder_name>` with your own name for experiment), twice each:

  - `python collect_maximization_images.py --dataset binary --seed_mode_normal True --model_type /p/adversarialml/as9rw/models_cifar10binary_vgg/cifar10_binary_10pdog_v2_linf/checkpoint.pt.best --save_path /p/adversarialml/jyc9fyf/<folder_name> --sample_ratio 0.1`
  
  - `python collect_maximization_images.py --dataset binary --seed_mode_normal True --model_type /p/adversarialml/as9rw/models_cifar10binary_vgg/cifar10_binary_50pdog_v2_linf/checkpoint.pt.best --save_path /p/adversarialml/jyc9fyf/<folder_name> --sample_ratio 0.1`


## For 9/23

- We'll be running the same process with 5% sample ratio (twice, for reproducability) across various models trained on different ratios. This data (via various ways) will then be used to train a meta classifier.

- Run the following experiments (replace `<folder_name>` with your own name for experiment), varying '0p_linf' in the path in [0p_linf, 10p_linf, 20p_linf, 30p_linf, 40p_linf, 50p_linf, 60p_linf, 70p_linf, 80p_linf, 90p_linf, 100p_linf] 

- `python collect_maximization_images.py --seed_mode_normal True --model_type /p/adversarialml/as9rw/new_exp_models/small/0p_linf/checkpoint.pt.best --save_path /p/adversarialml/jyc9fyf/small/<folder_name> --sample_ratio 0.05`

- `python collect_maximization_images.py --seed_mode_normal True --model_type /p/adversarialml/as9rw/new_exp_models/small/0p_linf_2/checkpoint.pt.best --save_path /p/adversarialml/jyc9fyf/small_2/<folder_name> --sample_ratio 0.05`


## For 10/13

Here's a brief tutorial to sklearn: [link](https://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting). It's quite straight forward: you can use `.predict()` on models to get predictions, `.fit()` to train, etc. The documentation is self-explanatory. Let me know if you have any doubts. 

1. Looking at instances where models trained on different datasets fail or are right: a qualitative analysis of those examples. The file `implems/functional_test.py` loads models (multiple trained on each dataset) and performs some analysis. You can look at the part relevant to getting model predictions [here](https://github.com/iamgroot42/fnb/blob/master/implems/functional_test.py#L70). 

2. You can access the initial layer (or any layer) of the model using `MLP.coefs_[0])` and `MLP.intercepts_[0]` (corresponding to the weight, bias) for any of those models. Some analysis on how each of them gives different importance (absolute, relative: we can look at both) might be useful. For instance, some of them might give 0 weight corresponding to the 'sex' feature, or have a high/low bias that corresponds to the classifier trained on an imbalanced-gender version of the dataset.

Note: You can use the mapping of column names [here](https://github.com/iamgroot42/fnb/blob/master/implems/functional_test.py#L36) to see what each feature actually corresponds to: for a better understanding of how different models are looking at those features differently.


## For 11/18

1. Re-run the functional tests we discussed in our meeting  (after fixing the filter-on-test-data bug that we went through in the meeting) and take note of trends/numbers for both the sex and gender filters.

2. Use `train_ratio_models.py` to train multiple models for different filters and ratios. For instance, to train 30 classifiers on data with the sex filter (60% females) and save them in your folder, run:

`python train_ratio_models.py --savepath MYFOLDER --filter sex --ratio 0.6 --num 30`

3. Run these experiments and save models in separate folders. Set `num` to 500 and run experiments for the following ratios and filters:

* sex: 0, 0.25, 0.4, 0.5, 0.6, 0.75, 1
* race: 0, 0.25, 0.4, 0.5, 0.6, 0.75, 1
* income: 0, 0.25, 0.4, 0.5, 0.6, 0.75, 1
* none: 0.5

4. Use `meta_classify.py` to train meta-classifiers using the models you trained above. This file takes as arguments paths to two folders, and takes models from them to train a meta classifier.
Example:

`python python meta_classify.py --path1 FOLDER1 --path2 FOLDER2 --sample SIZE`

Run these experiments for the following ratios:

* sex (0) vs sex (1)
* sex (0) vs sex (0.5)
* sex (0.5) vs sex (1)
* sex (0) vs sex (0.25)
* sex (0.25) vs sex (0.5)
* sex (0.5) vs sex (0.75)
* sex (0.75) vs sex (0.1)
* sex (0.4) vs sex (0.6)

* sex (0) vs normal
* sex (0.25) vs normal
* sex (0.4) vs normal
* sex (0.5) vs normal
* sex (0.6) vs normal
* sex (0.75) vs normal
* sex (1) vs normal


Repeat the above set of experiments for 'race' and 'income' instead of 'sex'. Also, for each experiment, vary `sample` in [10, 25, 50, 100, 150, 200, 300, 400, 500]
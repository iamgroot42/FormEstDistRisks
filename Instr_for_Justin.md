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

`python meta_classify.py --path1 FOLDER1 --path2 FOLDER2 --sample SIZE`

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


## For 1/5

### Task

Given a product review on Amazon, caegorize the associated rating as positive or negative. The property that we will be focussing on for now will be: <i>does the dataset contain product reviews for home-improvement related products (1) or not (0)?</i>

The dataset has been pre-processed with a <a href="https://huggingface.co/transformers/model_doc/roberta.html">RoBERT</a> model. Each product review corresponds to a n-dimensional feature vector, on top of which a 3-layer MLP is trained.

For this experiment, we want to know if the performance of meta-classifiers is because of them capturing inherent properties of these datasets indirectly, or just differentiating between "exact" datasets (not caring even if they have different properties).

Say M1 and M2 are trained on D1, D2 respectively (where D1 satisfies property, D2 does not). Currently, a meta-classifier can look at any model, say N, and tell if it was trained on a dataset that satisfied the property or not.

Problem is, in current experimental settings, this N itself is either trained or D1/D2, or trained on some dataset D3 that is highly overlapping with D1/D2, which might not happen in the real world. We want to see how the performance of a meta-classifier varies as we control this level of "overlap".

### Commands

* Navigate to `text` folder
* Make a folder to save trained models in
* Run the following script

  `bash in_line.sh <MODEL_PATH> <MERGE_RATIO> <0/1>`

  where:
  * first argument points to the folder where models will be saved
  * second argument is a ratio in [0, 1] that determines how much of the second dataset split will be used while training the models
  * the third argument determines if the dataset used will have the property (1) or not (0)

  For instance, training a model that uses 70% of the second dataset, without the property, saved in a folder names `MODELS`, the command would be:

  `bash in_line.sh MODELS/ 0.7 0`

* Run the above set of experiments while varying `<MERGE_RATIO>`: 0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9, for both `0` and `1` for the last argument. Don't forget to use different folders to save these models!

* Note: these scripts will take quite a while to run (a few hours ), so it might be in your best interest to use a screen/tmux to run them in. Also, depending on which GPU is free, you can add the `CUDA_VISIBLE_DEVICES` prefix. That way, you can run multiple scripts on a machine simultaneously, each on a different GPU.


## For 3/3

Train models for skin-lesion (HAM1000) dataset with two different ratios: 0.2 and 0.8, trying out both the 'sex' and 'age' attributes
We will train 3000 models each (which we will use for train set for the metaclassifier), and 500 models each (which we will use for test set for the meta classifier)

* Navigate to the appropriate folder (`ham/`)

* Run the following command to generate models (to be used to train)

`python train_models.py <attribute> <ratio> 3000 1 <save_dir>`


* Run the following command to generate models (to be used to test)

`python train_models.py <attribute> <ratio> 500 2 <save_dir>`


* `attribute` is `sex/age`, `ratio` is `0.2 or 0.8`, `save_dir` is where models will be saved. So overall, there wil be 2 (age/sex) * 2 (ratio 0.2/ratio 0.8) * 2 (first command/second command) = 8 scripts to run.

## For 4/7

I have updated the model to accept arguments that should enable retrieving intermediate model activations. For instance, model logits are usually retrieved via the command `outputs = model(images)[:, 0]`, which would return a `(n,)` shaped vector (where `n=images.shape[0]`).

To retreve activations, you will use `acts = model(images, latent=0)`, whch will return a `(n, num_features)` shaped vector. You can then count the number of fired neurons (value > 0) per image.

## For 16/7

This experiment is similar to the activations-based meta-classifier experiments we tried recently. Instead of using activations for existing samples, we will try generating samples that help maximize the difference between activation values for models from the two categories. We will also try sampling from existing data, and then compare the two methods.

- 3 trials of the following experiment, varying `second` in `[0.2, 0.3, 0.4, 0.6, 0.7, 0.8]`. Report train/test accuracies and plot the results. This version will use existing images and pick the one that maximize gap in activation values.

`python optimal_generation.py --n_samples 10 --second 0.8 --latent_focus 0 --n_models 20 --steps 500 --step_size 1e2 --use_natural`

- 3 trials of the following experiment, varying `second` in `[0.2, 0.3, 0.4, 0.6, 0.7, 0.8]`. Report train/test accuracies and plot the results. This one will generate inputs that maximize difference between activations for models from the two categories.

`python optimal_generation.py --n_samples 10 --second 0.8 --latent_focus 0 --n_models 20 --steps 500 --step_size 1e2`

## For 30/7

With this experiment, we want to figure out of minimal performance gains when using all layers emerge from redundancy in information stored in layers, or later layers actually not being very useful. For now, we will generate numbers on the BoneAge and Census (sex) datasets.

- For Census, run and store numbers, and generate box-plots (for visualization- preferably in the same graph. You can use seaborn) for the following experiments:
  - `python meta.py --start_n 1 --first_n 2 --filter sex`
  - `python meta.py --start_n 2 --first_n 3 --filter sex`
  - `python meta.py --start_n 3 --first_n 4 --filter sex`

- For BoneAge, same procedure as above (except that you will have to vary the ratios `X` in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8] yourself and store numbers):
  - `python meta.py --start_n 1 --first_n 2 --second X`
  - `python meta.py --start_n 2 --first_n 3 --second X`

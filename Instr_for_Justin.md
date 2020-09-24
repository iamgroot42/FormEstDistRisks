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

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

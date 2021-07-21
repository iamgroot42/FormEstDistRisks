Testing
=============
### Accuracy (test_plots_All.py)
Tests models by determining number of incorrect predictions on filtered data (sex = 0, 1 or age = 0, 1)

1. To setup an experiment, first set the variable "base" in the main method to the path where the HAM dataset is located and the variable "folder_paths" to the locations of the trained models. Both are shown below

	`base = "/p/adversarialml/as9rw/datasets/ham10000/" #dataset path `

        folder_paths = [
            "/u/jyc9fyf/hamModels/hamAge/testAge0.2",
            "/u/jyc9fyf/hamModels/hamAge/testAge0.8",
            "/u/jyc9fyf/hamModels/hamSex/testSex0.2",
            "/u/jyc9fyf/hamModels/hamSex/testSex0.8"
        ]

2. The functions process_data_sex(base, value) and process_data_age(base, value) are used to filter test data so that it only shows labels 1 or 0, which is indicated by the "value" argument. 
	Also, the function run_tests() runs the accuracy test once the data is processed. Below is an example of filtering data so that sex = 0 and running tests on the models in the folder paths:
	    df = process_data_sex(base,value = 0)
        incorrectSex0 = run_tests() #returns a list with the average number of errors in each model
	Processing and testing can be done multiple times if more data is needed.
3. Finally, the data is plotted in the form of a scatterplot and saved. The title and save path should also be changed accordingly.



### Activations (test_acts_models.py)
Tests models by determining the number of neuron activations (on filtered data)

1. To setup an experiment, first set the variable "base" in the main method to the path where the HAM dataset is located and the variable "folder_paths" to the locations of the trained models. Both are shown below

	`base = "/p/adversarialml/as9rw/datasets/ham10000/" #dataset path `

        folder_paths = [
            "/u/jyc9fyf/hamModels/hamAge/testAge0.2",
            "/u/jyc9fyf/hamModels/hamAge/testAge0.8",
            "/u/jyc9fyf/hamModels/hamSex/testSex0.2",
            "/u/jyc9fyf/hamModels/hamSex/testSex0.8"
        ]

2. The functions process_data_sex(base, value) and process_data_age(base, value) are used to filter test data so that it only shows labels 1 or 0, which is indicated by the "value" argument. 
	Also, the function run_tests() runs the activations test once the data is processed. Below is an example of filtering data so that sex = 1 and running tests on the models in the folder paths:
	    df = process_data_sex(base,value = 1)
        activate1,activate2,activate3,activate4 = run_tests() #returns a list with the total number of activations
	However, because of the way the code was written, a difference in activations testing is that run_tests() only tests the first model in each folder.

3. Finally, the data is plotted in the form of a histogram and saved. The title and save path should also be changed accordingly.

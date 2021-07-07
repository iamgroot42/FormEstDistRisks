import joblib
#data
test_feature_names = "1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_1", "3_2", "3_3", "4_1", "4_2", "4_3", "5_1", "5_2", "5_3", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "8_1", "8_2", "8_3", "9_1", "9_2", "9_3", "10_1", "10_2", "10_3", "11_1", "11_2", "11_3", "12_1", "12_2", "12_3", "13_1", "13_2", "13_3", "14_1", "14_2", "14_3", "15_1", "15_2", "15_3", "16_1", "16_2", "16_3", "17_1", "17_2", "17_3", "18_1", "18_2", "18_3", "19_1", "19_2", "19_3", "20_1", "20_2", "20_3"

from sklearn.datasets import load_iris
iris = load_iris()

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model_path = '/p/adversarialml/as9rw/models_celeba/75_25/adv/Male/0.0/100_0.9100060096153846_0.22157855704426765.pth'
model = joblib.load(model_path)
exit()

# Extract single tree
estimator = model.estimators_[5]
tree = "tree0.0"

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file=tree + '.dot', 
                feature_names = test_feature_names,
                class_names = test_feature_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', tree + '.dot', '-o', tree + '.png', '-Gdpi=600'], shell = True)

'''
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')'''
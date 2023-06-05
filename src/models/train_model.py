import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from model_utils import train_rf_with_pca
from model_utils import test_rf_with_PCA  
import matplotlib.pyplot as plt

# loading pickle file
with open("data/processed/01_data_feature.pkl", 'rb') as f:
    loaded_dictionary = pickle.load(f)

# creating feature inputs:

X_train_stat_features = loaded_dictionary['X_train_stat_features']
X_test_stat_features  = loaded_dictionary['X_test_stat_features']

X_train_psd_features = loaded_dictionary['X_train_psd_features']
X_test_psd_features  = loaded_dictionary['X_test_psd_features']

X_train_beat_features = loaded_dictionary['X_train_beat_features']
X_test_beat_features  = loaded_dictionary['X_test_beat_features']

y_train = loaded_dictionary['y_train']
y_test  = loaded_dictionary['y_test']


## Train a model

n_components =5
print('results for stat features')
rf, act1 = train_rf_with_pca(X_train_stat_features, y_train, PCA_T=True, nPCA=n_components,PCA_VAR_plot=True,gridsearch=True)
acte1 = test_rf_with_PCA(rf,X_test_stat_features,y_test,PCA_T=True,nPCA=n_components,PCA_VAR_plot=True)

print('results for psd features')
rf, act2= train_rf_with_pca(X_train_psd_features, y_train, PCA_T=False, nPCA=n_components,PCA_VAR_plot=False,gridsearch=True)
acte2 = test_rf_with_PCA(rf,X_test_psd_features,y_test,PCA_T=False,nPCA=n_components,PCA_VAR_plot=False)

print('results for beat features')
rf, act3 = train_rf_with_pca(X_train_beat_features, y_train, PCA_T=True, nPCA=n_components,PCA_VAR_plot=True,gridsearch=True)
acte3 = test_rf_with_PCA(rf,X_test_beat_features,y_test,PCA_T=True,nPCA=n_components,PCA_VAR_plot=True)


features = ['Statistical', 'Power Spectrum', 'Heart Beat']

accuracy = [[act1, acte1], [act2, acte2], [act3, acte3]]
accuracy = np.array(accuracy)
n_groups = len(features)

# Create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
rects1 = plt.bar(index, accuracy[:, 0], bar_width, color='b', label='Train')
rects2 = plt.bar(index + bar_width, accuracy[:, 1], bar_width, color='r', label='Test')

# Labeling
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.title('Accuracy by feature and by train/test')
plt.xticks(index + bar_width / 2, features)  # positioning of feature names at the x-axis
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

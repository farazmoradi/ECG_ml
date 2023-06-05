import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import pandas as pd


# open and load data from a pickle file
with open("data/interim/01_data_laoded.pkl", "rb") as file:
    loaded_data = pickle.load(file)

# extracting test and train
X_train = loaded_data['X_train']
X_test = loaded_data['X_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']

print('Number of training examples: ',len(X_train))
print('Number of test examples: ',len(X_test))

# Convert y_train to integer type
y_train = y_train.astype(int)

class_counts = np.bincount(y_train)

for class_label, count in enumerate(class_counts):
    if class_label==0:
        t=count
    print(f"Class {class_label} has {count} trials")
    if class_label!=0:
        print(f"Class {class_label} fraction to class 0 is {t/count}")

## Plotting the number of samples per each class for training and test set

arrhythmia_type = ['NOR', 'LBB', 'RBB', 'PVC', 'APC']
colorl= ['red','orange','blue','magenta','purple']

fig, ax = plt.subplots(1,2,figsize =(10,5),dpi=150)
# Calculate the count of each class
classes, counts = np.unique(y_train, return_counts=True)
ax[0].bar(arrhythmia_type, counts,color=colorl)

classes, counts = np.unique(y_test, return_counts=True)
ax[1].bar(arrhythmia_type, counts,color=colorl)

ax[0].set_title('Number of Training Examples per Class',fontsize=10)
ax[1].set_title('Number of Test Examples per Class',fontsize=10)
ax[0].set_ylabel('Count')
ax[1].set_ylabel('Count')
mpl.style.use("seaborn-v0_8-deep")
plt.tight_layout()
plt.savefig('reports/figures/comparing_train_test.jpg',dpi=200,facecolor='white')
plt.close()

################# The raw data 

fig1, ax1 = plt.subplots(len(arrhythmia_type),1,figsize =(5,5),dpi=150)
for i in range(len(arrhythmia_type)):
    ax1[i].plot(np.average(X_train[np.where(y_train==i)[0]],0),color=colorl[i])
    ax1[i].fill_between(np.arange(0,X_train.shape[1]), np.average(X_train[np.where(y_train==i)[0]],0) + np.std(X_train[np.where(y_train==0)[0]],0), np.average(X_train[np.where(y_train==i)[0]],0) - np.std(X_train[np.where(y_train==0)[0]],0),
        alpha=0.2, edgecolor=colorl[i], facecolor=colorl[i])
    ax1[i].set_title(arrhythmia_type[i])
    
ax1[i].set_xlabel('time points')
    
mpl.style.use("seaborn-v0_8-deep")
plt.tight_layout()
plt.savefig('reports/figures/comparing_average.jpg',dpi=200,facecolor='white')
plt.close()
    
    

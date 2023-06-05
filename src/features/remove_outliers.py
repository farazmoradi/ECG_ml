import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl


# loading pickle file
open_file = open("data/interim/01_data_laoded.pkl", "rb")
loaded_list = pickle.load(open_file)

# extracting test and train
X_train = loaded_list['X_train']
X_test = loaded_list['X_test']
y_train = loaded_list['y_train']
y_test = loaded_list['y_test']

# plotting the boxplot of average for 5 classes

# Set a custom color palette
sns.set_palette("Set2")

# Create a list of the five arrays
data = [np.mean(X_train[np.where(y_train==0)],1),np.mean(X_train[np.where(y_train==1)],1),
        np.mean(X_train[np.where(y_train==2)],1),np.mean(X_train[np.where(y_train==3)],1),
        np.mean(X_train[np.where(y_train==4)],1)]

# Increase figure size and DPI for better quality
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

# Create the box plot with improved aesthetics
arrhythmia_type = ['NOR', 'LBB', 'RBB', 'PVC', 'APC']
box_plot = ax.boxplot(data, patch_artist=True,labels=arrhythmia_type)

# Customize box colors
colors= ['red','orange','blue','magenta','purple']
for box, color in zip(box_plot['boxes'], colors):
    box.set(facecolor=color)

# Customize whisker, cap, and median colors
for element in ['whiskers', 'caps', 'medians']:
    plt.setp(box_plot[element], color='black')

# Customize flier properties
plt.setp(box_plot['fliers'], marker='o', markerfacecolor='red', markersize=6, linestyle='none', alpha=0.6)

# Customize the font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Box Plot of Five Arrhythmia Types', fontsize=14)

# Set a white grid
ax.grid(True)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 100
# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Load ECG data (replace with your own dataset)
ecg_signal = X_train[14,:]

# Check for consecutive zero values
missing_data_mask = np.concatenate(([False], np.diff(ecg_signal) == 0))

# Plot the ECG signal with missing data highlighted
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal, color='blue', label='ECG signal')
plt.plot(np.where(missing_data_mask)[0], ecg_signal[missing_data_mask], 'ro', markersize=3, label='Missing data')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('ECG Signal with Missing Data')
plt.legend()
plt.grid(True)
plt.show()



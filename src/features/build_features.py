import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import skew, kurtosis, percentileofscore
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from feature_functions import calculate_stat_features, calculate_single_beat_features,bandpower,add_noise,time_shift,scaling,spec_augment,spec_augment_ep



# loading pickle file
open_file = open("data/interim/01_data_laoded.pkl", "rb")
loaded_list = pickle.load(open_file)

# extracting test and train
X_train = loaded_list['X_train']
X_test = loaded_list['X_test']
y_train = loaded_list['y_train']
y_test = loaded_list['y_test']


##########################   interpolation methods for missing data in the future   ########################## 

# plt.plot(X_train[1,:])

# # Find indices of zeros
# zero_indices = np.where(X_train[1,:] == 0)[0]

# # Create an array of indices for interpolation
# interp_indices = np.arange(len(X_train[1,:]))

# # Perform linear interpolation
# X_train[1,:][zero_indices] = np.interp(interp_indices[zero_indices], interp_indices[~zero_indices], X_train[1,:][~zero_indices])

# # Print the interpolated ECG signal
# print(X_train[1,:])

##########################   extracting basic statistical features from ECG   ########################## 

X_train_stat_features = calculate_stat_features(X_train)
X_test_stat_features = calculate_stat_features(X_test)


##########################   extracting heart beat features from ECG   ########################## 

sampling_rate = 125
X_train_beat_features = []
X_test_beat_features = []

for i in range(len(X_train)):
    X_train_beat_features.append(list(calculate_single_beat_features(X_train[i,:], sampling_rate).values()))
    
for i in range(len(X_test)):
    X_test_beat_features.append(list(calculate_single_beat_features(X_test[i,:], sampling_rate).values()))
    
##########################   band-pass filtering for future projects if sampling rate was higher   ########################## 

# Band-pass filter data
#lowcut = 0.01
#highcut = 100
#order = 3
#ecg_filtered = butter_bandpass_filter(X_train[1,:], lowcut, highcut, sampling_rate, order)

##########################   calulating the average power of important frequnecy bands   ########################## 


X_train_psd_features = []
X_test_psd_features  = []

# Define frequency bands
bands = {'band1': [0.1, 5],
         'band2': [5, 10],
         'band3': [10, 15],
         'band4': [15, 20],
         'band5': [20, 25],
         'band6': [25, 30],
         'band7': [30, 35]}

# Compute the average power of each frequency band

for i in range(len(X_train)):
    X_train_psd_features.append(list({band: bandpower(X_train[i,:], sampling_rate, freqs) for band, freqs in bands.items()}.values()))
    
for i in range(len(X_test)):
    X_test_psd_features.append(list({band: bandpower(X_test[i,:], sampling_rate, freqs) for band, freqs in bands.items()}.values()))


##########################   exporting extracted features from the original dataset   ########################## 

open_file = open("data/processed/01_data_feature.pkl", "wb")

pickle.dump({'X_train_stat_features':X_train_stat_features,'X_test_stat_features':X_test_stat_features,
             'X_train_psd_features':X_train_psd_features,'X_test_psd_features':X_test_psd_features,
             'X_train_beat_features':X_train_beat_features,'X_test_beat_features':X_test_beat_features,
             'y_train':y_train,'y_test':y_test}, open_file)

with open('data/processed/01_data_feature.pkl', 'wb') as f:
    pickle.dump({'X_train_stat_features':X_train_stat_features,'X_test_stat_features':X_test_stat_features,
             'X_train_psd_features':np.array(X_train_psd_features),'X_test_psd_features':np.array(X_test_psd_features),
             'X_train_beat_features':np.array(X_train_beat_features),'X_test_beat_features':np.array(X_test_beat_features),
             'y_train':y_train,'y_test':y_test}, f)

print('All features are saved')




##########################   data augmentation   ########################## 


# Test the function with a random ECG signal
augmented_signal = spec_augment(X_train[1,:], Nf=100, Nt=200)

fig, ax = plt.subplots(2,1,sharex=True)

ax[0].plot(X_train[1,:])
ax[1].plot(augmented_signal)
ax[0].set_title('ECG data',fontsize=15)
ax[1].set_title('Augmented ECG data using SpecAugment',fontsize=15)
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = (20,5)
plt.tight_layout()
plt.savefig('reports/figures/spec_aug.jpg',dpi=200,facecolor='white')
plt.close()
    



       
X_train_spec_0 = spec_augment_ep(X_train,y_train,0,1)
X_train_spec_0 = X_train_spec_0[0:5000,:]
X_train_spec_1 = spec_augment_ep(X_train,y_train,1,6)
X_train_spec_2 = spec_augment_ep(X_train,y_train,2,4)
X_train_spec_3 = spec_augment_ep(X_train,y_train,3,8)
X_train_spec_4 = spec_augment_ep(X_train,y_train,4,4)

X_test_spec_0 = spec_augment_ep(X_test,y_test,0,1)
X_test_spec_0 = X_test_spec_0[0:1000,:]
X_test_spec_1 = spec_augment_ep(X_test,y_test,1,6)
X_test_spec_2 = spec_augment_ep(X_test,y_test,2,4)
X_test_spec_3 = spec_augment_ep(X_test,y_test,3,8)
X_test_spec_4 = spec_augment_ep(X_test,y_test,4,4)


X_train_add_noise_1_4 = add_noise(X_train[np.where(y_train!=0)])
y_train_add_noise_1_4  = y_train[np.where(y_train!=0)]

X_test_add_noise_1_4   = add_noise(X_test[np.where(y_test!=0)])
y_test_add_noise_1_4   = y_test[np.where(y_test!=0)]

X_train_time_shift_1_4   = time_shift(X_train[np.where(y_train!=0)])
y_train_time_shift_1_4   = y_train_add_noise_1_4

X_test_time_shift_1_4    = time_shift(X_test[np.where(y_test!=0)])
y_test_time_shift_1_4    = y_test_add_noise_1_4

X_train_scaling_1_4   = scaling(X_train[np.where(y_train!=0)])
y_train_scaling_1_4   = y_train_add_noise_1_4

X_test_scaling_1_4    = scaling(X_test[np.where(y_test!=0)])
y_test_scaling_1_4    = y_test_add_noise_1_4


X_train_add_noise_0 = add_noise(X_train[np.where(y_train==0)])
X_train_add_noise_0 = X_train_add_noise_0[0:5000,:]
y_train_add_noise_0 = np.zeros(X_train_add_noise_0.shape)

X_test_add_noise_0  = add_noise(X_test[np.where(y_test==0)])
X_test_add_noise_0  = X_test_add_noise_0[0:1000,:]
y_test_add_noise_0  = np.zeros(X_test_add_noise_0.shape)

X_train_time_shift_0  = time_shift(X_train[np.where(y_train==0)])
X_train_time_shift_0  = X_train_time_shift_0[0:5000,:]
y_train_time_shift_0  = y_train_add_noise_0

X_test_time_shift_0  = time_shift(X_test[np.where(y_test==0)])
X_test_time_shift_0  = X_test_time_shift_0[0:1000,:]
y_test_time_shift_0  = y_test_add_noise_0

X_train_scaling_0  = scaling(X_train[np.where(y_train==0)])
X_train_scaling_0  = X_train_scaling_0[0:5000,:]
y_train_scaling_0  = y_train_add_noise_0

X_test_scaling_0  = scaling(X_test[np.where(y_test==0)])
X_test_scaling_0  = X_test_scaling_0[0:1000,:]
y_test_scaling_0  = y_test_add_noise_0


## Combining Data

# train
X_train_aug_0 = np.concatenate([X_train_spec_0,X_train_add_noise_0,X_train_time_shift_0,X_train_scaling_0],axis=0)
y_train_aug_0 = np.zeros(X_train_aug_0.shape[0])


X_train_aug_1_4_spec = np.concatenate([X_train_spec_1,X_train_spec_2,X_train_spec_3,X_train_spec_4])
y_train_aug_1_4_spec = np.concatenate([np.ones(X_train_spec_1.shape[0]), 2*np.ones(X_train_spec_2.shape[0]),
                                       3*np.ones(X_train_spec_3.shape[0]), 4*np.ones(X_train_spec_4.shape[0])])


X_train_aug_1_4 = np.concatenate([X_train_aug_1_4_spec,X_train_add_noise_1_4,X_train_time_shift_1_4,X_train_scaling_1_4])
y_train_aug_1_4 = np.concatenate([y_train_aug_1_4_spec,y_train_add_noise_1_4,y_train_time_shift_1_4,y_train_scaling_1_4])

assert(X_train_aug_1_4.shape[0]==y_train_aug_1_4.shape[0])


# Final Concact
X_train_aug = np.concatenate([X_train_aug_0,X_train_aug_1_4_spec,X_train_aug_1_4])
y_train_aug = np.concatenate([y_train_aug_0,y_train_aug_1_4_spec,y_train_aug_1_4])

# test
X_test_aug_0 = np.concatenate([X_test_spec_0,X_test_add_noise_0,X_test_time_shift_0,X_test_scaling_0],axis=0)
y_test_aug_0 = np.zeros(X_test_aug_0.shape[0])


X_test_aug_1_4_spec = np.concatenate([X_test_spec_1,X_test_spec_2,X_test_spec_3,X_test_spec_4])
y_test_aug_1_4_spec = np.concatenate([np.ones(X_test_spec_1.shape[0]), 2*np.ones(X_test_spec_2.shape[0]),
                                       3*np.ones(X_test_spec_3.shape[0]), 4*np.ones(X_test_spec_4.shape[0])])


X_test_aug_1_4 = np.concatenate([X_test_aug_1_4_spec,X_test_add_noise_1_4,X_test_time_shift_1_4,X_test_scaling_1_4])
y_test_aug_1_4 = np.concatenate([y_test_aug_1_4_spec,y_test_add_noise_1_4,y_test_time_shift_1_4,y_test_scaling_1_4])

assert(X_test_aug_1_4.shape[0]==y_test_aug_1_4.shape[0])

# Final Conact
X_test_aug = np.concatenate([X_test_aug_0,X_test_aug_1_4_spec,X_test_aug_1_4])
y_test_aug = np.concatenate([y_test_aug_0,y_test_aug_1_4_spec,y_test_aug_1_4])

#######################################

arrhythmia_type = ['NOR', 'LBB', 'RBB', 'PVC', 'APC']
colorl= ['red','orange','blue','magenta','purple']

fig, ax = plt.subplots(1,2,figsize =(10,5))

# Calculate the count of each class
classes, counts = np.unique(y_train_aug, return_counts=True)
ax[0].bar(arrhythmia_type, counts,color=colorl)

classes, counts = np.unique(y_test_aug, return_counts=True)
ax[1].bar(arrhythmia_type, counts,color=colorl)

ax[0].set_title('Number of Training Examples per Class')
ax[1].set_title('Number of Test Examples per Class')
ax[0].set_ylabel('Count')
ax[1].set_ylabel('Count')

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 100
plt.tight_layout()
plt.savefig('reports/figures/aug_train.jpg',dpi=200,facecolor='white')
plt.close()


# Convert y_train to integer type
y_train_aug = y_train_aug.astype(int)

class_counts = np.bincount(y_train_aug)

for class_label, count in enumerate(class_counts):
    if class_label==0:
        t=count
    print(f"Class {class_label} has {count} trials")
    if class_label!=0:
        print(f"Class {class_label} fraction to class 0 is {t/count}")


# Export data: 

# Specify the filename in write binary("wb") mode
with open('data/processed/01_data_augmentation.pkl', 'wb') as f:
    pickle.dump({'X_train_aug':X_train_aug,'y_train_aug':y_train_aug,
                 'X_test_aug':X_test_aug,'y_test_aug':y_test_aug}, f)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import pickle
import matplotlib as mpl

# loading data

# loading pickle file
open_file = open("data/interim/01_data_laoded.pkl", "rb")
loaded_list = pickle.load(open_file)

# extracting test and train
X_train = loaded_list['X_train']
X_test = loaded_list['X_test']
y_train = loaded_list['y_train']
y_test = loaded_list['y_test']

# loading pickle file
open_file = open("data/processed/01_data_augmentation.pkl", "rb")
loaded_list = pickle.load(open_file)

# extracting test and train
X_train_aug = loaded_list['X_train_aug']
X_test_aug = loaded_list['X_test_aug']
y_train_aug = loaded_list['y_train_aug']
y_test_aug = loaded_list['y_test_aug']


sample_rate = 125

# Set STFT parameters
window_size = 48  # Window size of 20 milliseconds
hop_size = 40  # Hop size of 10 milliseconds

fig, ax = plt.subplots(2,1,sharex=True)

# Compute STFT
frequencies, times, magnitude_spectrogram = stft(X_train[0,:], fs=sample_rate,
                                                window='hann', nperseg=window_size, noverlap=hop_size)

im1 = ax[0].pcolormesh(times, frequencies, np.abs(magnitude_spectrogram),shading='auto')

frequencies, times, magnitude_spectrogram = stft(X_train_aug[0,:], fs=sample_rate,
                                                window='hann', nperseg=window_size, noverlap=hop_size)

im2 = ax[1].pcolormesh(times, frequencies, np.abs(magnitude_spectrogram),shading='auto')

ax[1].set_xlabel('Time (s)')
ax[0].set_ylabel('Frequency (Hz)')
ax[0].set_title('ECG Spectrogram')
ax[1].set_title('Augmented ECG SpecAug Spectrogram')
ax[1].set_ylabel('Frequency (Hz)')

fig.colorbar(im1, ax=ax[0])
fig.colorbar(im2, ax=ax[1])

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = (10,5)

plt.tight_layout()
plt.savefig('reports/figures/stft.jpg',dpi=200,facecolor='white')
plt.close()

# Converting all data to images


X_train_stft = []
X_test_stft  = []


# Set STFT parameters
window_size = 48  # Window size of 20 milliseconds
hop_size = 40  # Hop size of 10 milliseconds

for i in range(len(X_train)):

    frequencies, times, magnitude_spectrogram = stft(X_train[i,:], fs=sample_rate,
                                                window='hann', nperseg=window_size, noverlap=hop_size)
    X_train_stft.append(abs(magnitude_spectrogram))

for i in range(len(X_test)):

    frequencies, times, magnitude_spectrogram = stft(X_test[i,:], fs=sample_rate,
                                                window='hann', nperseg=window_size, noverlap=hop_size)
    X_test_stft.append(abs(magnitude_spectrogram))

X_train_stft = np.array(X_train_stft)
X_test_stft  = np.array(X_test_stft)


X_train_stft_aug = []
X_test_stft_aug  = []


# Set STFT parameters
window_size = 48  # Window size of 20 milliseconds
hop_size = 40  # Hop size of 10 milliseconds

for i in range(len(X_train_aug)):

    frequencies, times, magnitude_spectrogram = stft(X_train_aug[i,:], fs=sample_rate,
                                                window='hann', nperseg=window_size, noverlap=hop_size)
    X_train_stft_aug.append(abs(magnitude_spectrogram))

for i in range(len(X_test_aug)):

    frequencies, times, magnitude_spectrogram = stft(X_test_aug[i,:], fs=sample_rate,
                                                window='hann', nperseg=window_size, noverlap=hop_size)
    X_test_stft_aug.append(abs(magnitude_spectrogram))

X_train_stft_aug = np.array(X_train_stft_aug)
X_test_stft_aug  = np.array(X_test_stft_aug)

print(X_train_stft_aug.shape)

# Export data

X_train_stft = X_train_stft[:,:,:,np.newaxis]
X_test_stft = X_test_stft[:,:,:,np.newaxis]

# Specify the filename in write binary("wb") mode
with open('data/processed/01_data_stft.pkl', 'wb') as f:
    pickle.dump({'X_train':X_train_stft,'y_train':y_train,
                 'X_test':X_test_stft,'y_test':y_test}, f)
    
X_train_stft_aug = X_train_stft_aug[:,:,:,np.newaxis]
X_test_stft_aug  = X_test_stft_aug[:,:,:,np.newaxis]  
    
X_train_total = np.concatenate([X_train_stft,X_train_stft_aug])
X_test_total  = np.concatenate([X_test_stft,X_test_stft_aug])

y_train_total = np.concatenate([y_train,y_train_aug])
y_test_total  = np.concatenate([y_test,y_test_aug])

# Specify the filename in write binary("wb") mode
with open('data/processed/01_data_aug_stft.pkl', 'wb') as f:
    pickle.dump({'X_train':X_train_total,'y_train':y_train_total,
                 'X_test':X_test_total,'y_test':y_test_total}, f)
    
    
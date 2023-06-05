
import numpy as np
from scipy.stats import skew, kurtosis, percentileofscore
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import simps
from scipy.signal import stft, istft
import random

def calculate_stat_features(data):
    
## Input is a 2D ECG (trial*timepoints)    
    # Calculate statistical features
    mean     = np.mean(data,1)
    median   = np.median(data,1)
    std_dev  = np.std(data,1)
    maximum  = np.max(data,1)
    minimum  = np.mean(data,1)
    srange   = maximum - minimum
    skewness = skew(data,1)
    kurt    = kurtosis(data,1)
    Q1 = np.percentile(data, 25,axis=1)
    Q3 = np.percentile(data, 75,axis=1)
    IQR = Q3 - Q1

    features = np.concatenate([mean[:,np.newaxis],median[:,np.newaxis],std_dev[:,np.newaxis],maximum[:,np.newaxis],
                                    minimum[:,np.newaxis],srange[:,np.newaxis],skewness[:,np.newaxis],
                                    kurt[:,np.newaxis],Q1[:,np.newaxis],Q3[:,np.newaxis],IQR[:,np.newaxis]],axis=1)
    return features

def calculate_single_beat_features(heartbeat_signal, fs):
    # Parameters
    n = len(heartbeat_signal)
    duration = n / fs  # Duration of the heartbeat signal (in seconds)
    
    # Find R-peak amplitude and duration
    r_peak_amplitude = np.max(heartbeat_signal)
    r_peak_duration = n / fs
    
    # Find P-wave amplitude and duration
    p_wave_amplitude = np.max(heartbeat_signal[:int(n * 0.2)])  # Assuming P-wave is within the first 20% of the heartbeat
    p_wave_duration = (np.argmax(heartbeat_signal[:int(n * 0.2)]) / fs)  # Assuming P-wave is within the first 20% of the heartbeat
    
    # Find PR interval
    pr_interval = (np.argmax(heartbeat_signal[:int(n * 0.2)]) / fs)  # Assuming P-wave is within the first 20% of the heartbeat
    
    # Find QRS duration
    qrs_duration = duration
    
    # Find QT interval
    qt_interval = duration
    
    # Find ST segment
    st_segment = 0  # Assuming the ST segment is isoelectric
    
    # Find T-wave amplitude and duration
    t_wave_amplitude = np.max(heartbeat_signal[int(n * 0.6):])  # Assuming T-wave is within the last 40% of the heartbeat
    t_wave_duration = (np.argmax(heartbeat_signal[int(n * 0.6):]) / fs)  # Assuming T-wave is within the last 40% of the heartbeat
    
    return {
        'R-Peak Amplitude': r_peak_amplitude,
        'R-Peak Duration': r_peak_duration,
        'P-Wave Amplitude': p_wave_amplitude,
        'P-Wave Duration': p_wave_duration,
        'PR Interval': pr_interval,
        'QRS Duration': qrs_duration,
        'QT Interval': qt_interval,
        'ST Segment': st_segment,
        'T-Wave Amplitude': t_wave_amplitude,
        'T-Wave Duration': t_wave_duration
    }
    
# Butterworth Bandpass Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Bandpower function
def bandpower(data, sf, band):
    freqs, psd = welch(data, sf)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = simps(psd[idx_band], dx=freq_res)
    return bp

def add_noise(X_train, noise_factor=0.02):
    noise = np.random.randn(*X_train.shape) * noise_factor
    X_train_noisy = X_train + noise
    return X_train_noisy

def time_shift(X_train, shift_max=100):
    shift = random.randint(-shift_max, shift_max)
    return np.roll(X_train, shift, axis=1)

def scaling(X_train, scaling_factor=2):
    scaling_vector = np.random.uniform(low=1.0/scaling_factor, high=scaling_factor, size=(1,X_train.shape[1]))
    return X_train * scaling_vector



def spec_augment(signal, Nf, Nt, w=0.5, R=1):
    # Apply the Short-time Fourier transform (STFT) to the signal
    f, t, Zxx = stft(signal,fs=125)

    for _ in range(R):
        # Perform spectral masking
        Nm_f = int(w * Nf)
        start_f = np.random.randint(0, Nf - Nm_f)
        Zxx[start_f : start_f + Nm_f, :] = 0

        # Perform temporal masking
        Nm_t = int(w * Nt)
        start_t = np.random.randint(0, Nt - Nm_t)
        Zxx[:, start_t : start_t + Nm_t] = 0

    # Convert the masked STFT back to the time-domain
    _, augmented_signal = istft(Zxx)

    return np.append(augmented_signal,0)
        
        
 # augmentation    
        
def spec_augment_ep(data,label,n_class,n_augment):
    augment = []
    for i,j in enumerate(label):
        if j==n_class:
            for i in range(n_augment):
                augment.append(spec_augment(data[i,:], Nf=100, Nt=200))
                
    return np.array(augment)
import librosa
import numpy as np
import os



def calc_mfccs(audio_data, samplerate, n_mfcc_coeffs=13, fft_window_size=512, hop_length=160):
  mfcc = librosa.feature.mfcc(audio_data, sr=samplerate, n_mfcc=n_mfcc_coeffs, n_fft=fft_window_size, hop_length=hop_length)

  # add derivatives and normalize, the derivatives are concatinated exactly like this in the speecht code
  mfcc_delta = librosa.feature.delta(mfcc)
  mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
  mfcc = np.concatenate((normalize(mfcc),
                         normalize(mfcc_delta),
                         normalize(mfcc_delta2)), axis=0)

  return mfcc.T

def calc_power_spectrogram(audio_data, samplerate, n_mels=128, n_fft=512, hop_length=160):
  """
  Calculate power spectrogram from the given raw audio data

  Args:
    audio_data: numpyarray of raw audio wave
    samplerate: the sample rate of the `audio_data`
    n_mels: the number of mels to generate
    n_fft: the window size of the fft
    hop_length: the hop length for the window

  Returns: the spectrogram in the form [time, n_mels]

  """
  spectrogram = librosa.feature.melspectrogram(audio_data, sr=samplerate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

  # convert to log scale (dB)
  log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

  # normalize
  normalized_spectrogram = normalize(log_spectrogram)

  return normalized_spectrogram.T

def normalize(values):
  return (values - np.mean(values)) / np.std(values)

def waveform_file_to_mfcc(waveform_file_path, output_sample_length):
    waveform, sample_rate = librosa.load(waveform_file_path)

    # Resample to 16kHz for correct window size and hop length
    waveform = librosa.core.resample(waveform, sample_rate, 16000)
    sample_rate = 16000

    if np.size(waveform) > output_sample_length:
        waveform = waveform[1:output_sample_length+1]
    elif np.size(waveform) < output_sample_length:
        waveform = np.concatenate([waveform, waveform[1:output_sample_length-np.size(waveform)+1]])

    return calc_mfccs(waveform, sample_rate)

def waveform_file_to_power_spectrogram(waveform_file_path, output_sample_length):
    waveform, sample_rate = librosa.load(waveform_file_path)

    # Resample to 16kHz for correct window size and hop length
    waveform = librosa.core.resample(waveform, sample_rate, 16000)
    sample_rate = 16000

    if np.size(waveform) > output_sample_length:
        waveform = waveform[1:output_sample_length+1]
    elif np.size(waveform) < output_sample_length:
        waveform = np.concatenate([waveform, waveform[1:output_sample_length-np.size(waveform)+1]])

    return calc_power_spectrogram(waveform, sample_rate)

def convert_files(waveform_directory, power_spectrogram_directory):
    sub_dirs = os.listdir(waveform_directory)

    print(str(len(sub_dirs)) + ' subdirectories found.')

    buff_num = 1
    sub_dir_index = 0
    power_spectrogram_buff = []
    for sub_dir in sub_dirs:
        waveform_dir = waveform_directory + '/' + sub_dir + '/wav'

        if os.path.isdir(waveform_dir):
            if sub_dir_index % 100 == 0:
                print('Processing subdir ' + str(sub_dir_index) + '.')
            for waveform_file in os.listdir(waveform_dir):
                power_spectrogram_buff.append(waveform_file_to_power_spectrogram(waveform_dir + '/' + waveform_file, 16000*5))

                if len(power_spectrogram_buff) == 1000:
                    np.save(power_spectrogram_directory + '/power_batch_' + str(buff_num) + '.npy', power_spectrogram_buff)
                    power_spectrogram_buff = []
                    buff_num += 1

        sub_dir_index += 1
    if len(power_spectrogram_buff) > 0:
        np.save(power_spectrogram_directory + '/power_batch_' + str(buff_num) + '.npy', power_spectrogram_buff)


if __name__ == '__main__':
    #mfcc = waveform_file_to_mfcc('/home/karl/Documents/deep/project/test.wav', 16000*5)
    #print(mfcc.shape)
    convert_files('/media/karl/Elements/DeepLearningProject/VoxForge/waveform/Spanish', '/media/karl/Elements/DeepLearningProject/VoxForge/power/Spanish')
    #convert_files('VoxForge/waveform/French', 'VoxForge/mfcc/French')
    #convert_files('VoxForge/waveform/Spanish', 'VoxForge/mfcc/Spanish')

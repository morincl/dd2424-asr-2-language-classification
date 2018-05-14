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

def convert_files(waveform_directory, mfcc_directory):
    sub_dirs = os.listdir(waveform_directory)

    print(str(len(sub_dirs)) + ' subdirectories found.')

    buff_num = 1
    sub_dir_index = 0
    mfcc_buff = []
    for sub_dir in sub_dirs:
        waveform_dir = waveform_directory + '/' + sub_dir + '/wav'

        if os.path.isdir(waveform_dir):
            if sub_dir_index % 100 == 0:
                print('Processing subdir ' + str(sub_dir_index) + '.')
            for waveform_file in os.listdir(waveform_dir):
                #waveform_file_to_mfcc_file(waveform_dir + '/' + waveform_file, mfcc_directory + '/' + sub_dir + '_' + waveform_file, 16000*5)
                mfcc_buff.append(waveform_file_to_mfcc(waveform_dir + '/' + waveform_file, 16000*5))

                if len(mfcc_buff) == 1000:
                    np.save(mfcc_directory + '/mfcc_batch_' + str(buff_num) + '.npy', mfcc_buff)
                    mfcc_buff = []
                    buff_num += 1

        sub_dir_index += 1
    if len(mfcc_buff) > 0:
        np.save(mfcc_directory + '/mfcc_batch_' + str(buff_num) + '.npy', mfcc_buff)


if __name__ == '__main__':
    convert_files('VoxForge/waveform/English', 'VoxForge/mfcc/English')
    convert_files('VoxForge/waveform/French', 'VoxForge/mfcc/French')
    convert_files('VoxForge/waveform/Spanish', 'VoxForge/mfcc/Spanish')

import numpy as np

# Short command for converting our power spectrum format to the speechT format

def convert_file(input_file, output_file_base_path):
    transcript = np.array([0]) # required by speechT

    spectrogram_list = np.load(input_file)
    spectrogram_id = 0
    for spectrogram in spectrogram_list:
        np.savez(output_file_base_path + str(spectrogram_id) + '.npz', audio_fragments=spectrogram, transcript=transcript)
        spectrogram_id += 1

#TODO: extend script to convert all files in a directory so we can run a batch conversion job

input_path = '/media/karl/Elements/DeepLearningProject/VoxForge/power/English/power_batch_1.npy'
output_base_path = '/media/karl/Elements/DeepLearningProject/VoxForge/power-speechT/English/power_batch_1_'
convert_file(input_path, output_base_path)

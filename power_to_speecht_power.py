import numpy as np
import os

# Short command for converting our power spectrum format to the speechT format

def convert_file(input_file, output_file_base_path):
    transcript = np.array([0]) # required by speechT

    spectrogram_list = np.load(input_file)
    spectrogram_id = 0
    for spectrogram in spectrogram_list:
        np.savez(output_file_base_path + str(spectrogram_id) + '.npz', audio_fragments=spectrogram, transcript=transcript)
        spectrogram_id += 1

def convert_folder(input_folder, output_folder):
    files = os.listdir(input_folder)

    print("Found " + str(len(files)) + " files.")
    file_index = 0
    for input_file in files:
        print("Converting file " + str(file_index+1))
        convert_file(input_folder + '/' + input_file, output_folder + '/' + input_file[0:-4] + '_')
        file_index += 1

# input_path = '/media/karl/Elements/DeepLearningProject/VoxForge/power/English/power_batch_1.npy'
# output_base_path = '/media/karl/Elements/DeepLearningProject/VoxForge/power-speechT/English/power_batch_1_'
# convert_file(input_path, output_base_path)

input_folder = '/media/karl/Elements/DeepLearningProject/VoxForge/power/Spanish'
output_folder = '/media/karl/Elements/DeepLearningProject/VoxForge/power-speechT/Spanish'
convert_folder(input_folder, output_folder)

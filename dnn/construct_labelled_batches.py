import numpy as np
import os


input_data_folder_path = "/media/david/Elements/David/deep/auto-enc-output/auto-enc-output/"
# output_data_folder_path = "/home/david/Documents/deep/dd2424-asr-2-language-classification/dnn/data/labelled/"
output_data_folder_path = "/media/david/Elements/David/deep/final_data/labelled/"

# data = np.load(input_data_folder_path + "autoencoded_batch0_english.npy")

languages = {"English": np.asarray([1, 0]),
             "French": np.asarray([0, 1])
             }
data = []

number_of_batches = len(os.listdir(input_data_folder_path + "French"))

for language in languages:
    files = os.listdir(input_data_folder_path + language)
    for i in range(number_of_batches):
        print("Processing batch {} of {}".format(i, language))
        data_batch = np.load(input_data_folder_path + language + "/" + files[i])
        for sample_index in range(data_batch.shape[0]):
            data.append([data_batch[sample_index, :, :], languages[language]])

data = np.asarray(data)
np.random.shuffle(data)

for batch_number in range(number_of_batches * len(languages)):
    np.save(output_data_folder_path + "data_set_{}.npy".format(batch_number),
            data[batch_number * 1000: (batch_number + 1) * 1000])

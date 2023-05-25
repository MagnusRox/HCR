import pandas as pd
import model_settings as ms

dataset = pd.DataFrame(columns=['path','text'])


def process_file_list():
    directory = ms.input_orig_data_directory
    words_list = ms.input_word_list
    with open(words_list, "r") as f:
        all_lines = f.readlines()
        num_of_lines = len(all_lines)
        for each_line in all_lines:
            if each_line.startswith("#"):
                continue
            current_line_list = each_line.split(" ")
            file_name_split = current_line_list[0].split("-")
            current_line_sub_path = file_name_split[0] + "/" + file_name_split[0]+"-"+file_name_split[1]+"/"+current_line_list[0]+".png"
            complete_line_file_path = directory + current_line_sub_path
            dataset.loc[len(dataset.index)] = [complete_line_file_path,current_line_list[-1]]


process_file_list()
dataset.to_csv(ms.output_raw_data_csv)
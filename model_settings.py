import sys

input_orig_data_directory = sys.path[1] + "/DataSets/"
input_word_list = sys.path[1] + "/words.txt"

preprocessed_config = {
    "should_resize": True,
    "should_rotate": True,
    "should_saturate": True
}

preprocess_saturation_amount = [3, 6, 9, 12]
preprocess_angle_to_rotate = [10, 15, 20, 25, -10, -20, -30]

output_raw_data_csv = "data_raw.csv"
output_postprocessed_image_destination_path = "Post-processing/images/"
output_postprocessed_data = "Post-processing/processed_data.csv"
output_postprocessed_corruption_removed_data = 'Post-processing/processed-corrupt-removed.csv'

model_image_size = (360, 160)
model_config = {
    'batch': 64,
    'epochs': 30,
    'learning_rate': 0.001
}

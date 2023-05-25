import random
import pandas as pd
import tensorflow as tf
import os
import csv
import cv2
import model_settings as ms
from PIL import Image, ImageOps, ImageEnhance

class PreProcess:
    def __init__(self, raw_dataset_file, should_resize, should_rotate, should_saturate):
        self.percent_to_augment = .15
        self.raw_dataset = pd.read_csv(raw_dataset_file)
        self.destination_path = ms.output_postprocessed_image_destination_path
        self.processed_dataset = pd.DataFrame(columns=["path", "text"])
        try:
            if should_resize:
                self.resize_with_padding()
            if should_rotate:
                self.rotate_images()
            if should_saturate:
                self.increase_saturation()
        finally:
            self.processed_dataset.to_csv(ms.output_postprocessed_data)
            self.remove_corrupt_files()

    def resize_with_padding(self, expected_size=ms.model_image_size):
        for i in self.raw_dataset.index:
            try:
                img_path = self.raw_dataset["path"][i]
                if os.path.exists(img_path):
                    curImg = Image.open(img_path)
                    curImg.thumbnail((expected_size[0], expected_size[1]))
                    delta_width = expected_size[0] - curImg.size[0]
                    delta_height = expected_size[1] - curImg.size[1]
                    pad_width = delta_width // 2
                    pad_height = delta_height // 2
                    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
                    processed_img = ImageOps.expand(curImg, padding)
                    filepath = self.destination_path + img_path.split("/")[-1] + ".png"
                    processed_img.save(filepath)
                    self.processed_dataset.loc[len(self.processed_dataset.index)] = (
                    filepath, self.raw_dataset["text"][i])
            except:
                continue

    def increase_saturation(self):
        sampled_dataset = self.raw_dataset.sample(frac=self.percent_to_augment)
        for ind in sampled_dataset.index:
            img_path = sampled_dataset["path"][ind]
            saturation_amount = random.choice([3, 6, 9, 12])
            if os.path.exists(sampled_dataset["path"][ind]):
                try:
                    img = Image.open(img_path)
                    converter = ImageEnhance.Color(img)
                    saturated_image = converter.enhance(saturation_amount)
                    filepath = self.destination_path + img_path.split("/")[-1] + "_sat.png"
                    saturated_image.save(filepath)
                    self.processed_dataset.loc[len(self.processed_dataset.index)] = (
                        filepath, sampled_dataset["text"][ind])
                except:
                    continue

    def rotate_images(self):
        sampled_dataset = self.raw_dataset.sample(frac=self.percent_to_augment)
        for ind in sampled_dataset.index:
            angle_to_rotate = random.choice([10, 15, 20, 25, -10, -20, -30])
            img_path = sampled_dataset["path"][ind]
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    rotated_img = img.rotate(angle_to_rotate)
                    filepath = self.destination_path + img_path.split("/")[-1] + "_rot.png"
                    rotated_img.save(filepath)
                    self.processed_dataset.loc[len(self.processed_dataset.index)] = (
                        filepath, sampled_dataset["text"][ind])
                except:
                    continue

    def remove_corrupt_files(self):
        file = open(ms.output_postprocessed_corruption_removed_data, 'w', newline='')
        data = pd.read_csv(ms.output_postprocessed_data, index_col=False)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data['text'] = data['text'].str.lower()
        data = data.dropna()
        with file:
            header = ['path', 'text']
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            for ind in data.index:
                path = data['path'][ind]
                text = data['text'][ind]
                try:
                    image = tf.io.read_file(path)
                    image = tf.io.decode_image(image, channels=3, dtype=tf.float32)
                except tf.errors.InvalidArgumentError:
                    # Skip the file if it has an unknown image file format
                    print(f"Skipping file: {path} Unknown image file format.")
                    continue

                except Exception as e:
                    # Skip the file if any other exception occurs during reading or decoding
                    print(f"Skipping file: {path} Exception: {str(e)}")
                    continue

                try:
                    img = Image.open(path)
                    img.verify()  # to veify if its an img
                    img.close()  # to close img and free memory space
                    img = cv2.imread(path)
                    shape = img.shape
                except (IOError, SyntaxError) as e:
                    print('Bad file:', path)
                    continue
                except Exception as e:
                    print('Bad file: {}'.format(e), path)
                    continue
                writer.writerow({'path': path, 'text': text})


proc = PreProcess(ms.output_postprocessed_data,
                  ms.preprocessed_config["should_resize"],
                  ms.preprocessed_config["should_rotate"],
                  ms.preprocessed_config["should_saturate"]
                  )

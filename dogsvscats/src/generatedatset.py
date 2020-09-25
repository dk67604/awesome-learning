import os
import glob
import numpy as np
import tqdm
from pathlib import Path
import shutil
from argparse import ArgumentParser
import pprint
import random
import json


def num_files_in_folder(base_dir):
    """
        Method returns total number of files in base_dir
        :param base_dir:
        :return: number of files in directory
    """

    num_files = sum([len(files) for r, d, files in os.walk(base_dir)])
    return num_files


class Dataset:

    def __init__(self, arguments):
        self.images_path = arguments.images_path
        self.output_dir = arguments.output_dir
        self.training = "train"
        self.validation = "validation"
        self.sample = "sample"
        self.training_dir = None
        self.validation_dir = None
        self.sample_dir = None

    def create_folder_structure(self, base_dir, mode="train"):
        """

        :param base_dir:
        :param mode:
        :return:
        """
        if mode == 'train':
            mode = self.training
        elif mode == 'validation':
            mode = self.validation
        base_dir = base_dir + "/" + mode
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir)
        os.makedirs(base_dir + "/" + "cats")
        os.makedirs(base_dir + "/" + "dogs")
        return base_dir

    def sample_subset_folder_structure(self, base_dir):
        """

        :param base_dir:
        :return:
        """
        base_dir = base_dir + "/" + self.sample
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir)
        os.makedirs(base_dir + "/" + "train/cats")
        os.makedirs(base_dir + "/" + "train/dogs")
        os.makedirs(base_dir + "/" + "validation/cats")
        os.makedirs(base_dir + "/" + "validation/dogs")
        return base_dir

    def create_dogs_and_cats_dataset(self):
        """

        :return:
        """
        path = os.path.join(self.images_path, 'train')
        training_images = glob.glob(path + "/" + "*.jpg")
        print("Total images in dataset: {}".format(len(training_images)))
        # Training
        base_dir = self.create_folder_structure(self.output_dir, mode='train')
        self.training_dir = base_dir
        print("Training path: {}".format(base_dir))
        # Start copying files from training folder to respect cat and dog folder
        outer = tqdm.tqdm(total=len(training_images), desc='Files', position=0)
        for item in training_images:
            base_name = os.path.basename(item)
            if "cat" in base_name:
                shutil.copy(item, base_dir + "/" + "cats/")
            elif "dog" in base_name:
                shutil.copy(item, base_dir + "/" + "dogs/")
            outer.update(1)

        cat_images = glob.glob(self.training_dir + "/cats/" + "*.jpg")
        dog_images = glob.glob(self.training_dir + "/dogs/" + "*.jpg")
        # Validation
        base_dir = self.create_folder_structure(self.output_dir, mode='validation')
        self.validation_dir = base_dir
        print("Validation path: {}".format(base_dir))
        cat_validation_images = random.sample(cat_images, int(0.20 * len(cat_images)))
        outer = tqdm.tqdm(total=len(cat_validation_images), desc='Validation Cat images', position=0)
        for item in cat_validation_images:
            shutil.move(item, base_dir + "/" + "cats/")
            outer.update(1)

        dog_validation_images = random.sample(dog_images, int(0.20 * len(cat_images)))
        outer = tqdm.tqdm(total=len(dog_validation_images), desc='Validation Dog images', position=0)
        for item in dog_validation_images:
            shutil.move(item, base_dir + "/" + "dogs/")
            outer.update(1)

    def copy_images(self, cat_images, dog_images, mode):
        outer = tqdm.tqdm(total=len(cat_images), desc='Validation Cat images', position=0)
        for cat, dog in zip(cat_images, dog_images):
            shutil.copy(cat, self.sample_dir + "/" + mode + "/cats")
            shutil.copy(dog, self.sample_dir + "/" + mode + "/dogs")
            outer.update(1)

    def generate_sample(self):
        base_dir = self.sample_subset_folder_structure(self.output_dir)
        print("Sample dataset path: {}".format(base_dir))
        self.sample_dir = base_dir
        cat_images = glob.glob(self.training_dir + "/cats/" + "*.jpg")
        dog_images = glob.glob(self.training_dir + "/dogs/" + "*.jpg")
        if len(cat_images) == 0 or len(dog_images) == 0:
            print("Check training images exist")
            return
        sample_cat_image = random.sample(cat_images, 1000)
        sample_dog_image = random.sample(dog_images, 1000)

        self.copy_images(sample_cat_image, sample_dog_image, mode='train')
        cat_validation_images = glob.glob(self.validation_dir + "/cats/" + "*.jpg")
        dog_validation_images = glob.glob(self.validation_dir + "/dogs/" + "*.jpg")
        if len(cat_validation_images) == 0 or len(dog_validation_images) == 0:
            print("Check validation images exist")
            return
        sample_cat_image = random.sample(cat_validation_images, 500)
        sample_dog_image = random.sample(dog_validation_images, 500)
        self.copy_images(sample_cat_image, sample_dog_image, mode='validation')

    def create_dataset_summary(self,sample_mode=False):
        my_printer = pprint.PrettyPrinter()
        dataset_summary = dict()
        dataset_summary['training'] = {}
        dataset_summary['training']['path'] = self.training_dir
        dataset_summary['training']['total_images'] = num_files_in_folder(self.training_dir)
        dataset_summary['training']['cats'] = num_files_in_folder(self.training_dir + "/cats")
        dataset_summary['training']['dogs'] = num_files_in_folder(self.training_dir + "/dogs")
        dataset_summary['validation'] = {}
        dataset_summary['validation']['path'] = self.validation_dir
        dataset_summary['validation']['total_images'] = num_files_in_folder(self.validation_dir)
        dataset_summary['validation']['cats'] = num_files_in_folder(self.validation_dir + "/cats")
        dataset_summary['validation']['dogs'] = num_files_in_folder(self.validation_dir + "/dogs")
        if sample_mode:
            dataset_summary['sample'] = {}
            dataset_summary['sample']['path'] = self.sample_dir
            dataset_summary['sample']['total_images'] = num_files_in_folder(self.sample_dir)
            dataset_summary['sample']['train'] = num_files_in_folder(self.sample_dir + "/train")
            dataset_summary['sample']['validation'] = num_files_in_folder(self.sample_dir + "/validation")
        print("######################################################")
        print("################Dataset Summary#######################")
        my_printer.pprint(dataset_summary)
        print("####################End###############################")

        with open(os.path.join(self.output_dir, 'dataset.json'), 'w') as fp:
            json.dump(dataset_summary, fp, indent=4)


def run(arguments):
    """
    Driver for performing the task
    :param arguments: Command Line arguments
    :return: None
    """
    gen_dataset = Dataset(arguments)
    gen_dataset.create_dogs_and_cats_dataset()
    if arguments.sample:
        gen_dataset.generate_sample()
    gen_dataset.create_dataset_summary(arguments.sample)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--images-path", required=True, help="Path for Dogs and cats dataset")
    parser.add_argument("--output-dir", required=True, help="Path for generate dataset directory")
    parser.add_argument("--sample", action='store_true', help="Flag to set if sample dataset required")
    args = parser.parse_args()
    run(args)

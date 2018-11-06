import keras.backend as K
from keras.preprocessing import image
from config import *
import os
import numpy as np
import tensorflow as tf
import re

class load_data(object):
    def __init__(self, mode):
        """
        Initialize the load_data class

        Store the file_path, disparity_path and its mode ('train', 'test')
        :param mode:
        """
        self.file_path = config.FILE_PATH
        self.disparity_path = config.DISPARITY_PATH
        self.mode = mode

    def extract_list_of_file(self, txt_file_left, txt_file_right):
        """
        Extract files from .txt files

        :param txt_file_left: txt file that stores the file names of left images
        :param txt_file_right: txt file that stores the file names of right images
        :return: file names for left images and file names for right images
        """
        left = []
        right = []
        # left_write_file_name = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/left_list.txt"
        # right_write_file_name = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/right_list.txt"
        f = open(txt_file_left, 'r')
        g = open(txt_file_right, 'r')
        for filename in f:
            ls = filename.split()
            for i in ls:
                left.append(i)
        for filename in g:
            ls = filename.split()
            for i in ls:
                right.append(i)
        f.close()
        g.close()
        return left, right

    def image_loader_generator(self, file_list_left, file_list_right=None):
        """
        Image generator

        :param file_list_left: file list for left images
        :param file_list_right: file list for right images
        :return: (batch size, height, width, channels) images
        """
        L = len(file_list_left)
        while True:
            batch_start = 0
            batch_end = config.BATCH_SIZE
            while batch_start < L:
                limit = min(batch_end, L)
                if self.mode == 'test' and not config.DO_STEREO:
                    left_image = self.load_image_by_name(file_list_left[batch_start:limit])
                    yield left_image
                else:
                    left_image = self.load_image_by_name(file_list_left[batch_start:limit])
                    right_image = self.load_image_by_name(file_list_right[batch_start:limit])
                if self.mode == 'train':
                    # do_augment = K.random_uniform([], 0, 1)
                    # left_image_aug, right_image_aug = K.switch(do_augment > 0.5, self.data_augmentation(left_image, right_image), (left_image, right_image))
                    yield ([left_image, right_image])
                elif self.mode == 'test':
                    yield([left_image, right_image])

                batch_start += config.BATCH_SIZE
                batch_end += config.BATCH_SIZE

    def image_loader(self, file_path_left, file_path_right=None):
        """
        Load all images that stored in file path

        :param file_path_left: file path pointed to left images
        :param file_path_right: file path pointed to right images
        :return: shuffled [left images, right images] with shape (2, batch size, height, width, channels)
        """
        if self.mode == 'test' and not config.DO_STEREO:
            left_image = self.load_image_by_path(file_path_left)
            return left_image[0]/255
        else:
            left_image, right_image = self.load_image_by_path(file_path_left, file_path_right)
            left_image = left_image/255
            right_image = right_image/255
        if self.mode == 'train':
            min_after_dequeue = 200
            capacity = min_after_dequeue + 4 * config.BATCH_SIZE
            left_image_batch, right_image_batch = tf.train.shuffle_batch([left_image, right_image], config.BATCH_SIZE, capacity, min_after_dequeue, enqueue_many=True)

            return [left_image_batch, right_image_batch]
        elif self.mode == 'test':
            return [left_image[0], right_image[0]]

    def load_image_by_name(self, file_list):
        """
        We have file names of images, so we can extract images from the file_list

        :param file_list: list that stores the file names of images
        :return: Concatenated images
        """
        imgs = []
        for imgf in file_list:
            img_path = os.path.join(config.FILE_PATH, imgf)
            img = image.load_img(img_path, target_size=(128, 416))
            x = image.img_to_array(img, data_format='channels_last')
            x = np.expand_dims(x, axis=0)
            imgs.append(x)
        imgs = np.concatenate(imgs, axis=0)
        return imgs

    def load_image_by_path(self, left_file_path, right_file_path=None):
        """
        Since left images and right images have similar file name, we can use re.sub to find right images from left images.
        Each image has dim (128, 416)

        :param left_file_path: path that stores the left images
        :param right_file_path: path that stores the right images
        :return: left images and right images
        """
        limgs = []
        rimgs = []
        for filename in os.listdir(left_file_path):
            if right_file_path is None:
                img_path = os.path.join(left_file_path, filename)
                img = image.load_img(img_path, target_size=(128, 416))
                x = image.img_to_array(img, data_format='channels_last')
                x = np.expand_dims(x, axis=0)
                limgs.append(x)
            else:
                left_image_path = os.path.join(left_file_path, filename)
                right_filename = re.sub(r"(\d+)\_(\d+)\.(\w+)", r"\1_11.\3", filename)
                right_image_path = os.path.join(right_file_path, right_filename)
                limg = image.load_img(left_image_path, target_size=(128, 416))
                rimg = image.load_img(right_image_path, target_size=(128, 416))
                lx = image.img_to_array(limg, data_format='channels_last')
                rx = image.img_to_array(rimg, data_format='channels_last')
                lx = np.expand_dims(lx, axis=0)
                rx = np.expand_dims(rx, axis=0)
                limgs.append(lx)
                rimgs.append(rx)


        if right_file_path is None:
            limgs = np.concatenate(limgs, axis=0)
            return limgs
        else:
            limgs = np.concatenate(limgs, axis=0)
            rimgs = np.concatenate(rimgs, axis=0)
            return limgs, rimgs



    def data_augmentation(self, left_image, right_image):
        """
        1. Gamma 校正：
            对图像的gamma曲线进行编辑， 以对图像进行非线性色调编辑。提高图像对比度。

        2. 随机亮度：
            对图像进行随机亮度的改变

        3. 使图像饱和：
            对图像收敛至0-1（normalized 之后）

        :param left_image:
        :param right_image:
        :return:
        """
        random_gamma = K.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        random_brightness = K.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # saturate
        left_image_aug = K.clip(left_image_aug, 0, 1)
        right_image_aug = K.clip(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug
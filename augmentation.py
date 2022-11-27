import skimage.transform as transform
import numpy.random as random
import torch
import numpy as np


class Augmentation:
    @staticmethod
    def rotate_image(input, output, H=256, W=256):
        """
        Applies rotation to the image randomly choosing the angle.
        It requires gray-scaled images as input
        :param input: the input gray-scaled image
        :param output: mask containing all the features
        :return: the rotated input and output with shape (H, W)
        """
        angle = random.randint(0, 360)
        transf_input = transform.rotate(input.reshape(H, W), angle)
        transf_output = transform.rotate(output.reshape(H, W), angle)
        return transf_input, transf_output

    @staticmethod
    def similarity_transform_image(input, output, H=256, W=256):
        """
        Performs similarity transformation on a gray-scaled image
        :param input: gray-scale image
        :param output: mask containing all the features
        :param H: height of the image
        :param W: width of the image
        :return: the transformed input and output with shape (H, W)
        """
        ax_0 = np.random.randint(0, H/4)
        ax_1 = np.random.randint(0, W/4)
        #print(f"X-axis translation = {ax_0}, Y-axis translation = {ax_1}")
        scale_factor = np.random.uniform(low=0.7, high=1.3)
        #print(f"Scale factor = {scale_factor}")
        transf_matrix = transform.SimilarityTransform(scale=scale_factor, translation=(ax_0, ax_1))
        transf_input = transform.warp(input.reshape(H, W), transf_matrix)
        transf_output = transform.warp(output.reshape(H, W), transf_matrix)
        return transf_input, transf_output
        

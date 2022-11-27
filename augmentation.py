import skimage.transform as transform
import numpy.random as random
import torch
import numpy as np


class Augmentation:
    @staticmethod
    def rotate_image(input, output, H=256, W=256, angle=None):
        """
        Applies rotation to the image randomly choosing the angle.
        It requires gray-scaled images as input
        :param input: the input gray-scaled image
        :param output: mask containing all the features
        :return: the rotated input and output with shape (H, W)
        """
        if angle is None:
            angle = random.randint(-90, 90)
        transf_input = transform.rotate(input.reshape(H, W), angle)
        transf_output = transform.rotate(output.reshape(H, W), angle)
        return transf_input, transf_output

    @staticmethod
    def similarity_transform_image(input, output, H=256, W=256, point=None):
        """
        Performs similarity transformation on a gray-scaled image
        :param input: gray-scale image
        :param output: mask containing all the features
        :param H: height of the image
        :param W: width of the image
        :return: the transformed input and output with shape (H, W)
        """
        if point is None:
            ax_0 = np.random.randint(-H/4, H/4)
            ax_1 = np.random.randint(-H/4, W/4)
        else:
            ax_0 = point[0]
            ax_1 = point[1]
        transf_matrix = transform.SimilarityTransform(translation=(ax_0, ax_1))
        transf_input = transform.warp(input.reshape(H, W), transf_matrix)
        transf_output = transform.warp(output.reshape(H, W), transf_matrix)
        return transf_input, transf_output
        

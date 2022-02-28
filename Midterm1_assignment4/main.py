# The dataset includes original images as well as their semantic segmentation in 9 object classes (i.e. the image files
# whose name ends in “_GT”, where each pixel has a value which is the identifier of the semantic class associated to it).
# Each file has a name starting with a number from 1 to 8, which indicates the thematic subset of the image, followed by
# the rest of the file name. This thematic subset can be used for instance as a class for the full image in image
# classification tasks.
#
# Assignment 4 Perform image segmentation on all images in the dataset, using the normalized cut algorithm run on the
# top of superpixels rather than on raw pixels. For each image compute a performance metric (which one it is up to you
# to decide) measuring the overlap between the image segments identified by NCUT and the ground truth semantic
# segmentation.
# You do not need to show this metric for all images, rather focus on on selecting and discussing 2 examples of images
# that are well-segmented Vs 2 examples of images that are badly segmented (according to the above defined metric).
#
# Hint: in Python, you have an NCut implementation in the scikit-image library; in Matlab, you can use the original NCut
# implementation here. Superpixels are implemented both in Matlab as well as in OpenCV. Feel free to pickup the
# implementation you liked most (and motivate the choice).
#

import scikit-image

if __name__ == '__main__':
    None
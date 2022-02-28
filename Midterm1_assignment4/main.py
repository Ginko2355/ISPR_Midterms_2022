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
import numpy as np
from skimage import io,data, segmentation, color, metrics, measure
from skimage.future import graph
from matplotlib import pyplot as plt

img = io.imread("dataset/1_8_s.bmp")
img_grnd_truth = io.imread("dataset/1_8_s_GT.bmp")
mask_gt = measure.label(img_grnd_truth)
#img = data.coffee()
number_of_superpixels = 400
superpixel_compactness = 30
superpixel_mask = segmentation.slic(img,
                                    compactness=superpixel_compactness,
                                    n_segments=number_of_superpixels,
                                    start_label=1)

# Computing an image where each superpixel has as a color, the mean of the colors
# in the original image of each superpixel region:
superpixeled_img = color.label2rgb(superpixel_mask,
                                   img,
                                   kind='avg',
                                   bg_label=0)

# Region Adiacency Graph:
# Each node in the RAG represents a set of pixels within image with the same label in labels.
region_graph = graph.rag_mean_color(image=img,
                                    labels=superpixel_mask,
                                    mode='similarity')
# Applying the NCUT algorithm to the RAG:
ncut_mask = graph.cut_normalized(superpixel_mask, region_graph)

# Computing an image where each n-cutted region has as a color, the mean of the colors
# of the n-cutted region in the original image:
ncut_image = color.label2rgb(ncut_mask, img, kind='avg', bg_label=0)

#TODOaa: Stampare le mask e vedere che contengono, sopratutto quella
# della ground truth, cercare di avere delle mask uguale per compararle
# con variation_of_information

raise Exception("VEDI IL TODO")
print("Shape gt: ", np.shape(img_grnd_truth),
      " Ncut shape: ", np.shape(ncut_mask),
      " Sp mask shape: ", np.shape(superpixel_mask))
str_sim = metrics.variation_of_information(mask_gt,ncut_mask)
print("Structural similarity:", str_sim)

#Subplotting
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(superpixeled_img.astype('uint8'))
ax[1].imshow(ncut_image.astype('uint8'))

for a in ax:
    a.axis('off')

plt.show()

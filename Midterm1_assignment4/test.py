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


def get_gt_label_from_image(image_gt):
    gt_mask = np.zeros(shape=(np.shape(image_gt)[0],np.shape(image_gt)[1]),dtype="uint8")
    rgb_values_map = {(0,0,0) : 0}
    id = 1 # 0 remains 0

    for px in range(len(gt_mask)):
        for py in range(len(gt_mask[px])):
            rgb_triple = (image_gt[px][py][0],
                          image_gt[px][py][1],
                          image_gt[px][py][2])
            if rgb_triple not in rgb_values_map.keys():
                rgb_values_map[rgb_triple] = id
                id += 1

            gt_mask[px][py] = rgb_values_map[rgb_triple]

    print(np.unique(gt_mask))
    return gt_mask

def spixel_ncut(img_path,gt_path,number_of_superpixels,superpixel_compactness):
    img = io.imread(img_path)
    img_grnd_truth = io.imread(gt_path)
    mask_gt = get_gt_label_from_image(img_grnd_truth)
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
    gt_image = color.label2rgb(mask_gt, img, kind='avg', bg_label=0)

    splits, merges = metrics.variation_of_information(mask_gt, ncut_mask)

    #TODOO write image with variation of information on top as the metrics


if __name__ == '__main__':


    for i,j in zip([10,100,20,200,30,300],[10,100,20,200,30,300]):
        print(i, j)

    raise Exception("")


    img = io.imread("dataset/5_29_s.bmp")
    img_grnd_truth = io.imread("dataset/5_29_s_GT.bmp")
    mask_gt = get_gt_label_from_image(img_grnd_truth)
    number_of_superpixels = 400
    superpixel_compactness = 40
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
    gt_image = color.label2rgb(mask_gt, img, kind='avg', bg_label=0)

    splits, merges = metrics.variation_of_information(mask_gt, ncut_mask)
    print("I want False splits and False merges as low as possible, together.")
    print(f'False Splits: {splits}')
    print(f'False Merges: {merges}')

    #Subplotting
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

    textstr = f'Number of superpixel = ' + str(number_of_superpixels) + '\n\n' + \
              f'Superpixel compactness = ' + str(superpixel_compactness) + '\n\n' + \
              f'False splits = ' + str(splits) + '\n\n' + \
              f'False merges = ' + str(merges)

    ax[0][0].imshow(superpixeled_img.astype('uint8'))
    ax[0][0].set_title("Super pixeled image")
    ax[0][1].text(0.05, 0.75, textstr, horizontalalignment='left',verticalalignment='top',
                  bbox=dict(facecolor='wheat', alpha=0.5))
    ax[1][0].imshow(ncut_image.astype('uint8'))
    ax[1][0].set_title("N-cutted image")
    ax[1][1].imshow(gt_image.astype('uint8'))
    ax[1][1].set_title("Ground truth")


    for i in range(len(ax)):
        for a in ax[i]:
            a.axis('off')

    fig.tight_layout()
    plt.show()

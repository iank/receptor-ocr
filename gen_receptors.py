#!/usr/bin/env python
import cv2
import sys
import os

def load_images(im_directory, label_filename):
    images = {}
    label_file = open(label_filename)
    for line in label_file:
        filename, label = line.strip().split(',')
        label = label if label else ' '
        img = cv2.imread(os.path.join(im_directory, filename))
        if (label in images):
            images[label].append(img)
        else:
            images[label] = [img]
    return images

def print_frequency_table(images):
    table = [(x, len(images[x])) for x in sorted(images.keys())]
    print("\n".join(["{0}: {1}".format(*x) for x in table]))

# Usage: label_seg.py directory/ labels.txt
# where labels is "filename,label" each line

images = load_images(sys.argv[1], sys.argv[2])
print_frequency_table(images)


# TODO: center all images and pad
# TODO: generate random receptor field
# TODO: compute entropies for receptors
# TODO: prune receptor field
# TODO: save receptor field

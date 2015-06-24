#!/usr/bin/env python
import cv2
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def _get_centroid(img):
    h,w,d = img.shape
    im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = cv2.moments(im2)
    return (m['m10']/m['m00'], m['m01']/m['m00'])

def _get_diagonal(img):
    h,w,d = img.shape
    return np.sqrt(h**2 + w**2)

def _draw_receptor(img, receptor):
    x0, y0 = _get_centroid(img)
    diag = _get_diagonal(img)
    
    # compute receptor center
    x = (receptor['center'][0] - 0.5) * diag + x0
    y = (receptor['center'][1] - 0.5) * diag + y0
    length = diag * receptor['length']
    angle = receptor['angle']

    pt1 = (int(x + np.cos(angle)*length/2), int(y + np.sin(angle)*length/2))
    pt2 = (int(x - np.cos(angle)*length/2), int(y - np.sin(angle)*length/2))

    cv2.line(img, pt1, pt2, (0, 255, 0), 1)
    return img

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

def gen_receptors(n):
    # Generate receptors on a normalized space
    # image centroid is 0.5, 0.5. Receptors are defined
    # by a center, angle, and length (normalized 1 = image diagonal)
    # receptors stretching beyond the boundary of the image are okay.

    # Length: Rayleigh distributed. sigma = 0.08
    # Center: 2-D gaussian N([0.5,0.5], 0.2)
    # and angle uniform [0,pi)

    receptors = []
    for i in range(0,n):
        receptor = {
            'center': np.random.normal(0.5, 0.15, 2),
            'length': np.random.rayleigh(0.08),
            'angle': np.random.uniform(0, np.pi),
        }

        bound = lambda n: max(min(n, 1), 0)
        receptor['center'][0] = bound(receptor['center'][0])
        receptor['center'][1] = bound(receptor['center'][1])
        receptors.append(receptor)
    return receptors

# Usage: label_seg.py directory/ labels.txt
# where labels is "filename,label" each line

images = load_images(sys.argv[1], sys.argv[2])
print_frequency_table(images)
receptors = gen_receptors(1000)

img = images['1'][0]
for r in receptors:
    img = _draw_receptor(img, r)

plt.imshow(img, interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.savefig('/var/www/vara/rec.png', bbox_inches='tight')


# TODO: compute receptor activation for all images
# TODO: compute entropies for receptors
# TODO: prune receptor field
# TODO: save receptor field

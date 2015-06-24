#!/usr/bin/env python
from __future__ import division
import cv2
import cv2.cv as cv
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def _saveplot(img, filename):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.savefig('/var/www/vara/{0}.png'.format(filename), bbox_inches='tight')

# useful in info theory to let 0log0 = 0
def _ilog(x):
    if (x==0):
        return 0
    return np.log2(x)


# Binary entropy function
def _h_(x):
    return -1*(x*_ilog(x) + (1-x)*_ilog(1-x))

def _receptor_endpoints(desc, receptor):
    x0, y0 = desc['centroid']
    diag = desc['diag']
    img = desc['image']
    
    # compute receptor center
    x = (receptor['center'][0] - 0.5) * diag + x0
    y = (receptor['center'][1] - 0.5) * diag + y0
    length = diag * receptor['length']
    angle = receptor['angle']

    pt1 = (int(x + np.cos(angle)*length/2), int(y + np.sin(angle)*length/2))
    pt2 = (int(x - np.cos(angle)*length/2), int(y - np.sin(angle)*length/2))
    return (pt1,pt2)

# Is receptor activated by img?
def _p_activated(desc, receptor):
    pt1,pt2 = _receptor_endpoints(desc, receptor)
    li = cv.InitLineIterator(cv.fromarray(desc['image']), pt1, pt2)

    activation = 0
    pts = 0
    for px in li:
        activation += px
        pts += 1

    if (pts > 0):
        return activation/pts
    else:
        return activation

def _get_centroid(img):
    m = cv2.moments(img)
    return (m['m10']/m['m00'], m['m01']/m['m00'])

def _get_diagonal(img):
    h,w = img.shape
    return np.sqrt(h**2 + w**2)

def _draw_receptors(desc, receptors):
    img_draw = desc['instance'].copy()

    for receptor in receptors:
        pt1, pt2 = _receptor_endpoints(desc, receptor)
        color = _p_activated(desc, receptor)
        cv2.line(img_draw, pt1, pt2, color, 1)
    return img_draw

def _prior_px(images):
    num_instances = sum([len(images[x]) for x in images.keys()])

    # Computer p(X=x) prior
    px = {}
    for letter in images.keys():
        px[letter] = len(images[letter]) / num_instances
    return px

def _receptor_activation(images, receptor):
    p1_x = {}
    py = 0
    num_instances = 0
    for letter in images.keys():
        p1_x[letter] = 0
        for instance in images[letter]:
            num_instances += 1
            if (_p_activated(instance, receptor)):
                p1_x[letter] += 1
                py += 1
        p1_x[letter] /= len(images[letter])
    py /= num_instances
    return (p1_x, py)

# Find H(Y|X) = average of H(Y|X=x) = h(p(Y=1|X=x))
def _conditional_entropyYX(p1_x):
    HYX = 0
    for letter in p1_x.keys():
        HYX += _h_(p1_x[letter])
    HYX /= len(images.keys())
    return HYX

# Find H(X|Y=1) = -1*sum_x(p(X=x|Y=1))
def _conditional_entropyXY(px_1):
    HXY = 0
    for letter in px_1.keys():
        HXY += px_1[letter]*_ilog(px_1[letter])
    HXY *= -1
    return HXY

def compute_usefulness(images, receptor):
    px = _prior_px(images)
    p1_x, py = _receptor_activation(images, receptor)

    if (py == 0):
        return 0 # receptor is not useful if it is -never- on. HYX high, HXY 0

    # Find p(X=x|Y=1) = p(Y=1|X=x) * p(X=x) / P(Y=y)  -- (Bayes)
    px_1 = {}
    for letter in images.keys():
        px_1[letter] = p1_x[letter] * px[letter] / py

    # Find H(Y|X) = average of H(Y|X=x) = h(p(Y=1|X=x))
    HYX = _conditional_entropyYX(p1_x)
    HXY = _conditional_entropyXY(px_1)

    # HXY: How well receptor splits letter space (maximize this)
    # HYX: consistent receptors across variants of symbol minimize this
    usefulness = HXY * (1 - HYX)
    return usefulness

def load_images(im_directory, label_filename):
    images = {}
    label_file = open(label_filename)
    for line in label_file:
        filename, label = line.strip().split(',')
        label = label if label else ' '
        img = cv2.imread(os.path.join(im_directory, filename))
        im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        im2 = cv2.adaptiveThreshold(im2, 1, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY, 11, 2)
        im2 = 1 - im2

        desc = {
            'image': im2,
            'centroid': _get_centroid(im2),
            'diag': _get_diagonal(im2),
        }

        if (label in images):
            images[label].append(desc)
        else:
            images[label] = [desc]
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


if __name__ == "__main__":
    # Usage: label_seg.py directory/ labels.txt
    # where labels is "filename,label" each line

    images = load_images(sys.argv[1], sys.argv[2])
    #print_frequency_table(images)
    receptors = gen_receptors(5000)

    usefulness = [0]*len(receptors)
    for k,receptor in enumerate(receptors):
        us = compute_usefulness(images, receptor)
        print("Receptor {0} useful {1}".format(k,us))
        usefulness[k] = us

    receptor_field = np.zeros((len(receptors), 5))
    for k,receptor in enumerate(receptors):
        receptor_field[k][0] = receptor['center'][0]
        receptor_field[k][1] = receptor['center'][1]
        receptor_field[k][2] = receptor['length']
        receptor_field[k][3] = receptor['angle']
        receptor_field[k][4] = usefulness[k]

    receptor_field = receptor_field[receptor_field[:,4].argsort()[::-1]]
    np.save('receptor_field.npy', receptor_field)
        
    #img = images['f'][0]

    #img_draw = _draw_receptors(img, receptors)
    #plt.imshow(img_draw, interpolation='bicubic')
    #plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    #plt.savefig('/var/www/vara/rec.png', bbox_inches='tight')

    # TODO: prune receptor field
    # TODO: script to load receptor field and display it on arbitrary image
    # TODO: continuous generalization
    # TODO: generate training data for model: load receptor field, compute
    # activations for all images & associate w/ class label

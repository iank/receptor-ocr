from __future__ import division

import cv2
import cv2.cv as cv

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

def draw_receptors(desc, receptors):
    img_draw = desc['instance'].copy()

    for receptor in receptors:
        pt1, pt2 = _receptor_endpoints(desc, receptor)
        color = p_activated(desc, receptor)
        cv2.line(img_draw, pt1, pt2, color, 1)
    return img_draw

def saveplot(img, filename):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.savefig('/var/www/vara/{0}.png'.format(filename), bbox_inches='tight')

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
def p_activated(desc, receptor):
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

def save_field(receptors, usefulness, filename):
    receptor_field = np.zeros((len(receptors), 5))
    for k,receptor in enumerate(receptors):
        receptor_field[k][0] = receptor['center'][0]
        receptor_field[k][1] = receptor['center'][1]
        receptor_field[k][2] = receptor['length']
        receptor_field[k][3] = receptor['angle']
        receptor_field[k][4] = usefulness[k]

    receptor_field = receptor_field[receptor_field[:,4].argsort()[::-1]]
    np.save(filename, receptor_field)

def load_field(filename):
    receptors = np.load(filename)
    # TODO
#    return receptors, usefulness

from __future__ import division

import os
import cv2
import cv2.cv as cv

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

# Symmetric Kullback Leibler divergence between PDFs p and q, defined as
# D_{KL}(P||Q) + D_{KL}(Q||P)
def sym_kl_div(p, q):
    return _kl_div(p,q) + _kl_div(q,p)

def _kl_div(p,q):
    eps = 0.0001 # div by zero hack
    return sum(p*np.log((p+eps)/(q+eps)))

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
            if (p_activated(instance, receptor)):
                p1_x[letter] += 1
                py += 1
        p1_x[letter] /= len(images[letter])
    py /= num_instances
    return (p1_x, py)

def _h_(x):
    return -1*(x*_ilog(x) + (1-x)*_ilog(1-x))

def _ilog(x):
    if (x==0):
        return 0
    return np.log2(x)

# Find H(Y|X) = average of H(Y|X=x) = h(p(Y=1|X=x))
def _conditional_entropyYX(p1_x):
    HYX = 0
    for letter in p1_x.keys():
        HYX += _h_(p1_x[letter])
    HYX /= len(p1_x.keys())
    return HYX

# Find H(X|Y=1) = -1*sum_x(p(X=x|Y=1))
def _conditional_entropyXY(px_1):
    HXY = 0
    for letter in px_1.keys():
        HXY += px_1[letter]*_ilog(px_1[letter])
    HXY *= -1
    return HXY

def compute_activation(images, receptor):
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

    # HXY: How well receptor splits letter space (minimize this)
    # HYX: consistent receptors across variants of symbol minimize this
    receptor['HYX'] = HYX
    receptor['HXY'] = HXY
    receptor['px_1'] = px_1
    receptor['p1_x'] = p1_x

    C = len(receptor['px_1'].keys())
    max_hxy = -1*_ilog(1/C)

    # Maximize divergence from existing set, minimize uncertainty per-pattern
    # and across patterns
    usefulness = (max_hxy - receptor['HXY']) * (1 - receptor['HYX'])
    receptor['usefulness'] = usefulness

    return receptor

def _get_centroid(img):
    m = cv2.moments(img)
    return (m['m10']/m['m00'], m['m01']/m['m00'])

def _get_diagonal(img):
    h,w = img.shape
    return np.sqrt(h**2 + w**2)

def load_image(filename):
    img = cv2.imread(filename)
    im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im2 = cv2.adaptiveThreshold(im2, 1, cv2.ADAPTIVE_THRESH_MEAN_C, \
        cv2.THRESH_BINARY, 11, 2)
    im2 = 1 - im2

    desc = {
        'image': im2,
        'centroid': _get_centroid(im2),
        'diag': _get_diagonal(im2),
    }
    return desc

def draw_receptors_activated(desc, receptors):
    img_draw = desc['image'].copy()

    for receptor in receptors:
        pt1, pt2 = _receptor_endpoints(desc, receptor)
        color = p_activated(desc, receptor)
        if (color > 0.01):
            cv2.line(img_draw, pt1, pt2, 4, 1)
        else:
            cv2.line(img_draw, pt1, pt2, 2, 1)
    return img_draw

def draw_receptors(desc, receptors, annotations):
    img_draw = desc['image'].copy()

    for k,receptor in enumerate(receptors):
        pt1, pt2 = _receptor_endpoints(desc, receptor)
        color = 1
        cv2.line(img_draw, pt1, pt2, color, 1)
        cv2.putText(img_draw, annotations[k], pt2, cv2.FONT_HERSHEY_PLAIN, 1, color+1)
    return img_draw


def save_plot(img, filename):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.savefig(filename, bbox_inches='tight')

def load_images(im_directory, label_filename):
    images = {}
    label_file = open(label_filename)
    for line in label_file:
        filename, label = line.strip().split(',')
        label = label if label else ' '
        desc = load_image(os.path.join(im_directory, filename))

        if (label in images):
            images[label].append(desc)
        else:
            images[label] = [desc]
    return images

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

def save_field(receptors, filename):
    receptor_field = np.zeros((len(receptors), 4))
    for k,receptor in enumerate(receptors):
        receptor_field[k][0] = receptor['center'][0]
        receptor_field[k][1] = receptor['center'][1]
        receptor_field[k][2] = receptor['length']
        receptor_field[k][3] = receptor['angle']

    np.save(filename, receptor_field)

def load_field(filename):
    receptor_field = np.load(filename)
    receptors = []
    for k in range(receptor_field.shape[0]):
        receptor = {
            'center': (receptor_field[k,0], receptor_field[k,1]),
            'length': receptor_field[k,2],
            'angle': receptor_field[k,3],
        }
        receptors.append(receptor)
        
    return receptors

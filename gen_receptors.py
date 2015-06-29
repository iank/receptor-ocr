#!/usr/bin/env python
from __future__ import division
import cv2
import sys
import os
import numpy as np
import receptor_common as rc
import time

# useful in info theory to let 0log0 = 0

# Binary entropy function

def compute_usefulness(incl, rec, num):
    if 'diiv' not in rec:
        rec['diiv'] = 1

    p = np.array([rec['px_1'][k] for k in sorted(rec['px_1'])])
    q = np.array([incl['px_1'][k] for k in sorted(incl['px_1'])])
    rec['diiv'] += rc.sym_kl_div(p,q)

    C = len(rec['px_1'].keys())
    max_hxy = -1*rc._ilog(1/C)

    # Maximize divergence from existing set, minimize uncertainty per-pattern
    # and across patterns
    usefulness = rec['diiv']/num * (max_hxy - rec['HXY']) * (1 - rec['HYX'])
    rec['usefulness'] = usefulness
    return rec

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

    images = rc.load_images(sys.argv[1], sys.argv[2])
    #print_frequency_table(images)
    receptors = gen_receptors(5000)

    for k,receptor in enumerate(receptors):
        print("Computing activation for {0}".format(k))
        receptors[k] = rc.compute_activation(images, receptor)

    # Discard receptors that are never activated by training set
    receptors = [x for x in receptors if x != 0]

    S = []
    remaining = receptors
    remaining = sorted(remaining, key=lambda k: k['usefulness'])
    S.append(remaining.pop())
    while remaining:
        for k,receptor in enumerate(remaining):
            remaining[k] = compute_usefulness(S[-1], receptor, len(S))

        # pick most useful receptor, add to S
        remaining = sorted(remaining, key=lambda k: k['usefulness'])
        S.append(remaining.pop())
        print("S: {0} ({1})".format(len(S), time.time()))

    rc.save_field(S, 'receptor_field.npy')

    # TODO: continuous generalization
    # TODO: train NN, backpropagate receptor activations for each letter

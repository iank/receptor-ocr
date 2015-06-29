#!/usr/bin/env python
import receptor_common as rc
import sys

# Usage: activations.py directory/ labels.txt field.npy idx1 [idx2 .. [idxN]]

images = rc.load_images(sys.argv[1], sys.argv[2])
receptors = rc.load_field(sys.argv[3])
idxs = [int(x) for x in sys.argv[4:]]

for k in idxs:
    receptors[k] = rc.compute_activation(images, receptors[k])
    receptor = receptors[k]

    # I want p(Y|X=x) for all X
    print("{0}: avg in-symbol uncertainty H(Y|X)  : {1}".format(k, receptor['HYX']))
    print("{0}: Across-symbol uncertainty H(X|Y=1): {1}".format(k, receptor['HXY']))
    print('p(Y=1|X=x):')

    for label in receptor['p1_x']:
        print("\tp(Y=1|X='{0}'): {1}".format(label, receptor['p1_x'][label]))

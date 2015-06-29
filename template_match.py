#!/usr/bin/python
from __future__ import print_function
import receptor_common as rc
import sys
import numpy as np

# Usage:
# gen_training_csv.py training_directory/ labels.txt

images = rc.load_images(sys.argv[1], sys.argv[2])

labels = {}
k = 1

print("Class labels:", file=sys.stderr)
for label in images.keys():
    print("{0}: {1}".format(k, label), file=sys.stderr)
    labels[label] = k
    k += 1

for label in images.keys():
    for desc in images[label]:
        print(",".join([str(x) for x in desc['image'].flatten()]) + "," + str(labels[label]))

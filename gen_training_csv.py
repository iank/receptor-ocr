#!/usr/bin/python
from __future__ import print_function
import receptor_common as rc
import sys

# Usage:
# gen_training_csv.py training_directory/ labels.txt receptor_field.npy N

images = rc.load_images(sys.argv[1], sys.argv[2])
receptors = rc.load_field(sys.argv[3])
N = int(sys.argv[4])

labels = {}
k = 1

print("Class labels:", file=sys.stderr)
for label in images.keys():
    print("{0}: {1}".format(k, label), file=sys.stderr)
    labels[label] = k
    k += 1

for label in images.keys():
    for desc in images[label]:
        activation = []
        for receptor in receptors[:N]:
            activation.append(str(rc.p_activated(desc, receptor)))
        print(",".join(activation) + "," + str(labels[label]))

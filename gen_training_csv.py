#!/usr/bin/python
import receptor_common as rc
import sys

# Usage:
# gen_training_csv.py training_directory/ labels.txt receptor_field.npy

images = rc.load_images(sys.argv[1], sys.argv[2])
receptors = rc.load_field(sys.argv[3])

labels = {}
k = 1

print("Class labels:")
for label in images.keys():
    print("{0}: {1}".format(k, label))
    labels[label] = k
    k += 1

print("\n\nData\n\n")
for label in images.keys():
    for desc in images[label]:
        activation = []
        for receptor in receptors:
            activation.append(str(rc.p_activated(desc, receptor)))
        print ",".join(activation) + "," + str(labels[label])

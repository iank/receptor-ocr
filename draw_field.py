#!/usr/bin/env python
import cv2
import receptor_common as rc
import sys

# Usage: draw_field.py field.npy image.png output.png [idx1 [idx2 [...]]]
# Draw selected receptors from field.npy on image.png
receptors = rc.load_field(sys.argv[1])

desc = rc.load_image(sys.argv[2])
filename = sys.argv[3]
idxs = [int(x) for x in sys.argv[4:]]
img = rc.draw_receptors_activated(desc, [receptors[i] for i in idxs])

rc.save_plot(img, filename)

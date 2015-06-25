#!/usr/bin/env python
import cv2
import receptor_common as rc
import sys

# Usage: draw_field.py field.npy image.png N output.png
# Draw N most useful receptors from field.npy on image.png
receptors = rc.load_field(sys.argv[1])

desc = rc.load_image(sys.argv[2])

N = int(sys.argv[3])
img = rc.draw_receptors_activated(desc, receptors[:N])
rc.save_plot(img, sys.argv[4])

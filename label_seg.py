#!/usr/bin/env python
import cv2
import sys
import os

# Usage: label_seg.py directory/

print('filename,char')
for file in os.listdir(sys.argv[1]):
    img = cv2.imread(os.path.join(sys.argv[1], file))
    cv2.imshow('dst_rt', img)
    x = cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('%s,%s' % (file, chr(x)))

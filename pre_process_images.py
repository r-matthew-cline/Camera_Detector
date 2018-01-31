##############################################################################
##
## pre_process_images.py
##
## @author: Matthew Cline
## @version: 20180129
##
## Description: Script to resize all of the images to a common size and aspect
##
##############################################################################

import cv2
import sys
import os

input_dir = os.path.normpath(sys.argv[1])
output_dir = os.path.normpath(sys.argv[2])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

for imgName in images:
    try:
        inputFN = os.path.join(input_dir, imgName)
    except:
        print("Problem joining the input path for image %s" % imgName)
        sys.exit(1)
    try:
        img = cv2.imread(inputFN)
    except:
        print("Problem reading the image %s" % imgName)
        sys.exit(1)
    try:
        img = cv2.resize(img, (256,256))
    except:
        print("Problem resizing the image %s" % imgName)
        sys.exit(1)
    try:
        outputFN = os.path.join(output_dir, imgName)
    except:
        print("Problem creating the output path for %s" % imgName)
        sys.exit(1)
    try:
        cv2.imwrite(outputFN, img)
    except:
        print("Problem saving the image %s" % imgName)
        sys.exit(1)



import numpy as np
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#from pyimagesearch.shapedetector import ShapeDetector


class ShapeDetector:
    def __init__(self):
    	pass
    
    def detect(self, c):
    	# initialize the shape name and approximate the contour
    	shape = "unidentified"
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.04 * peri, True)




def main():

    #--------------------------------------------------------------------------
    # Load the raw images
    #--------------------------------------------------------------------------
    # Number of images to clean
    n_clean = 5
    # Threshold to delete small contours
    countour_thresh = 50.0
    # Padding for the bounding boxes
    padding = 3

    for i in range(n_clean):

        raw_file    = './raw_img/train_img_%s.png' % i
        image       = cv2.imread(raw_file)

        # Convert back to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold
        ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

        #--------------------------------------------------------------------------
        # Delete small contours
        #--------------------------------------------------------------------------

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # initialize the mask that will be used to remove the bad contours
        mask = np.ones(image.shape[:2], dtype="uint8") * 255
        areas = []

        # loop over the contours
        for c in cnts:
            # compute the area of the contour
            area = cv2.contourArea(c)

            # if the contour area is too small, draw it on the mask
            if area < countour_thresh:
                cv2.drawContours(mask, [c], -1, 0, -1)

            areas.append(area)

        # remove the small contours from the image 
        image = cv2.bitwise_and(image, image, mask=mask)

        # The number contours should be the top 3 largest contours in the image
        top_areas   = sorted(areas, reverse=True)[0:3]
        digits      = [c for c in cnts if cv2.contourArea(c) in top_areas]

        #--------------------------------------------------------------------------
        # Create bounding boxes around 3 largest contours
        #--------------------------------------------------------------------------
        bboxes = []
        for c in digits:
            # Compute the bounding box for the contour
            x, y, w, h = cv2.boundingRect(c)

            # Save the bounding box (add 3 pixels
            bboxes.append((y-padding, y+h+padding, x-padding, x+w+padding))
 
        #--------------------------------------------------------------------------
        # invert the image
        #--------------------------------------------------------------------------
        image = ~image
 
        #--------------------------------------------------------------------------
        # Save the 3 bounding boxes as separate images
        #--------------------------------------------------------------------------
        j = 0
        for (ymin, ymax, xmin, xmax) in bboxes:

            single_digit    = image[ymin:ymax, xmin:xmax]
            digit_file      = "./digits/train_img%s_digit%s.png" % (i,j)

            cv2.imwrite(digit_file, single_digit)
            j = j + 1

        #--------------------------------------------------------------------------
        # save the clean image
        #--------------------------------------------------------------------------
        clean_file  = './clean_img/train_img_%s.png' % i 
        cv2.imwrite(clean_file, image)

        print("Clean image is in %s and the associated digits are inside the digit/ folder" % (clean_file))


if __name__ == "__main__":
    main()


# _Train_And_Test.py
"""Train a KNN model and test it."""

import operator
import os
import shutil
from datetime import datetime
import cv2
import numpy as np

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 180
MIN_CONTOUR_HEIGHT = 40
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################


class ContourWithData():
    """Character contour class.
    This contour frames each character in a given image.
    The character in each contour will be identified by the KNN model"""
    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        """calculate bounding rect info."""
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        """Checks if a contour is valis (has an area greater than the minimum contour area).
        This is oversimplified, for a production grade program
        much better validity checking would be necessary."""
        if self.intRectHeight < MIN_CONTOUR_HEIGHT:
            return False
        return True

###################################################################################################


def renameLatestImage():
    """Renames latest image to current timestamp."""
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M")
    latestImagePath = os.getcwd() + r'\latestImage.jpg'
    renamedLastImage = dt_string + '.jpg'
    renamedLatestImagePath = os.getcwd() + "\\" + renamedLastImage
    os.rename(latestImagePath, renamedLatestImagePath)
    return renamedLastImage


def main():
    """Train a KNN model using the following training data set:
    Features: flattened_images.txt
    Labels: classifications.txt
    Test KNN model using test image: """

    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=no-member

    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        # read in training classifications
        npaClassifications = np.loadtxt(
            "./training data/classifications.txt", np.float32)
    except FileNotFoundError:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        # read in training images
        npaFlattenedImages = np.loadtxt(
            "./training data/flattened_images.txt", np.float32)
    except FileNotFoundError:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))

    knn = cv2.ml.KNearest_create()                   # instantiate KNN object

    knn.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # read in testing numbers image
    image = cv2.imread("./latestImage.jpg")

    if image is None:                           # if image was not read successfully
        # print error message to std out
        print("error: image not read from file \n\n")
        # pause so user can see error message
        os.system("pause")
        # and exit function (which exits program)
        return
    # end if

    # get grayscale image
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)                    # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(
        # input image
        imgBlurred,
        # make pixels that pass the threshold full white
        255,
        # use gaussian rather than mean, seems to give better results
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        # invert so foreground will be white, background will be black
        cv2.THRESH_BINARY_INV,
        # size of a pixel neighborhood used to calculate threshold value
        11,
        # constant subtracted from the mean or weighted mean
        2)

    # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    imgThreshCopy = imgThresh.copy()

    # Divide the image to a grid of digits
    cv2.rectangle(
            # draw rectangle on original image
            image,
            # upper left corner
            (0, 100),
            # lower right corner
            (100,0),
            # green
            (0, 255, 0),
            # thickness
            2
        )

    # input image
    # make sure to use a copy
    # since the function will modify this image in the course of finding contours
    npaContours, _ = cv2.findContours(
        imgThreshCopy,
        # retrieve the outermost contours only
        cv2.RETR_EXTERNAL,
        # compress horizontal, vertical, and diagonal segments and leave only their end points
        cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:                             # for each contour
        # instantiate a contour with data object
        contourWithData = ContourWithData()
        # assign contour to contour with data
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(
            contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight(
        )                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(
            contourWithData.npaContour)           # calculate the contour area
        # add contour with data object to list of all contours with data
        allContoursWithData.append(contourWithData)
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            # if so, append to valid contour list
            validContoursWithData.append(contourWithData)
        # end if
    # end for

    validContoursWithData.sort(key=operator.attrgetter(
        "intRectX"))         # sort contours from left to right

    # declare final string, this will have the final number sequence by the end of the program
    strFinalString = ""

    for contourWithData in validContoursWithData:            # for each contour
        # draw a green rect around the current char
        cv2.rectangle(
            # draw rectangle on original image
            image,
            # upper left corner
            (contourWithData.intRectX, contourWithData.intRectY),
            # lower right corner
            (contourWithData.intRectX + contourWithData.intRectWidth,
             contourWithData.intRectY + contourWithData.intRectHeight),
            # green
            (0, 255, 0),
            # thickness
            2
        )

        # crop char out of threshold image
        imgROI = imgThresh[
            contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
            contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth
        ]

        # resize image, this will be more consistent for recognition and storage
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)

        _, npaResults, _, _ = knn.findNearest(
            npaROIResized, k=1)     # call KNN function find_nearest

        # get character from results
        strCurrentChar = str(chr(int(npaResults[0][0])))

        # append current char to full string
        strFinalString = strFinalString + strCurrentChar
    # end for

    print("\n" + strFinalString + "\n")                  # show the full string

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # show input image with green boxes drawn around found digits
    cv2.imshow("image", image)

    # wait for user key press
    cv2.waitKey(0)

    cv2.destroyAllWindows()             # remove windows from memory

    # renamedLastImage = renameLatestImage()
    # shutil.move('./' + renamedLastImage, './image archive/' + renamedLastImage)

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if

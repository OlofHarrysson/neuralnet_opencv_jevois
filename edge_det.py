import libjevois as jevois
import cv2
import numpy as np

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module by default simply converts the input image to a grayscale OpenCV image, and then applies the Canny
# edge detection algorithm. Try to edit it to do something else (note that the videomapping associated with this
# module has grayscale image outputs, so that is what you should output).
#
# @author Laurent Itti
#
# @displayname Python OpenCV
# @videomapping GRAY 640 480 20.0 YUYV 640 480 20.0 JeVois PythonOpenCV
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PythonOpenCV:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("canny", 100, jevois.LOG_INFO)

    # ###################################################################################################
    ## Process function with no USB output
    def process(self, inframe):
        jevois.LFATAL("process with no USB output not implemented yet in this module")

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured):
        inimg = inframe.get()
        #jevois.LINFO("Input image is {} {}x{}".format(jevois.fccstr(inimg.fmt), inimg.width, inimg.height))

        # Start measuring image processing time:
        self.timer.start()

        # Convert the input image to OpenCV grayscale:
        inimggray = jevois.convertToCvGray(inimg);

        # We are done with the input image:
        inframe.done()

        # Get the next available USB output image:
        outimg = outframe.get()
        #jevois.LINFO("Output image is {} {}x{}".format(jevois.fccstr(outimg.fmt), outimg.width, outimg.height))

        # Require that output image has same dims as input and is grayscale:
        outimg.require("output", inimg.width, inimg.height, jevois.V4L2_PIX_FMT_GREY);

        # Detect edges using the Canny algorithm from OpenCV:
        edges = cv2.Canny(inimggray, 100, 200, apertureSize = 3)

        # Copy the edge map to the output image to send over USB. Since here both are gray, this will be a simple copy:
        jevois.convertCvGRAYtoRawImage(edges, outimg, 100)

        # Write frames/s info from our timer:
        fps = self.timer.stop()
        jevois.writeText(outimg, fps, 3, outimg.height-13, jevois.YUYV.White, jevois.Font.Font6x10)

        # We are done with the output, ready to send it to host over USB:
        outframe.send()

    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    def parseSerial(self, str):
        return "ERR: Unsupported command"

    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    def supportedCommands(self):
        return ""
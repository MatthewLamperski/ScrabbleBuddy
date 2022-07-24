from visualization import VideoOCROutputBuilder
from blur_detector import detect_blur_fft
from imutils.video import VideoStream
from imutils.perspective import four_point_transform
from pytesseract import Output
import pytesseract
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to optional input video (webcam will be used otherwise)")
ap.add_argument("-o", "--output", type=str, help="path to optional output video")
ap.add_argument("-c", "--min-conf", type=int, default=50, help="minimum confidence value to filter weak text detection")
args = vars(ap.parse_args())

# initializations
outputBuilder = None
writer = None
outputW = None
outputH = None

# create named window for OCR visualization
cv2.namedWindow("Output")

# should webcam be used?
webcam = not args.get("input", False)

if webcam:
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

while True:
    # grab frame
    orig = vs.read()
    orig = orig if webcam else orig[1]

    # if reading from file and eof is reached, end
    if not webcam and orig is None:
        break

    # resize frame, compute ratio of new width to old width
    frame = imutils.resize(orig, width=600)
    ratio = orig.shape[1] / float(frame.shape[1])

    if outputBuilder is None:
        outputBuilder = VideoOCROutputBuilder(frame)

    # initialize card/ocr outputs
    card = None
    ocr = None

    # convert to grayscale and detect if considered blurry or not
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, thresh=15)

    # add text to represent blurry
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # only do expensive ocr stuff if not blurry
    if not blurry:
        # blur image and do edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cv2.imshow("Edged", edged)

        # find contours in edgemap and sort by size (desc), keeping only large ones
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # initialize contour that corresponds to the business card
        cardCnt = None

        # loop over contours
        for c in cnts:
            # approximate contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if contour has four points, we have found outline of card
            if len(approx) == 4:
                cardCnt = approx

                break

        if cardCnt is not None:
            # draw outline of business card on frame to verify
            cv2.drawContours(frame, [cardCnt], -1, (0, 255, 0), 3)

            # apply perspective transform
            card = four_point_transform(orig, cardCnt.reshape(4, 2) * ratio)

            color = (0, 255, 255)
            text = "CARD FOUND"
            cv2.putText(frame, text, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # allocate mem for OCR visualization
            ocr = np.zeros(card.shape, dtype="uint8")

            # swap channel ordering for card and OCR it
            rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
            cv2.imshow("RGB", rgb)
            results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

            # loop over each text localization
            for i in range(0, len(results["text"])):

                # extract bounding box coords
                x = results["left"][i]
                y = results["top"][i]
                w = results["width"][i]
                h = results["height"][i]

                # get ocr text itself
                text = results["text"][i]
                print(results["text"][i])
                conf = int(float(results["conf"][i]))

                # filter out weak confidence text, draw borderbox and text itself
                if conf > args["min_conf"]:
                    if len(text) > 0:
                        cv2.rectangle(card, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(ocr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    output = outputBuilder.build(frame, card, ocr)

    # check if video writer is none and output video file path was supplied
    if args["output"] is not None and writer is None:
        # grab output dims and init writer
        (outputH, outputW) = output.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 27, (outputW, outputH), True)

    # if writer already init'd write output to disk
    if writer is not None:
        # force resize video visualization to match dimensions of output video
        outputFrame = cv2.resize(output, (outputW, outputH))
        writer.write(outputFrame)

    # show output
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if webcam:
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()

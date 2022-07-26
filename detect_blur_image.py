from blur_detector import detect_blur_fft
import numpy as np
import argparse
import imutils
import cv2

# construct arg parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path input image")
ap.add_argument("-t", "--thresh", type=int, default=20, help="threshold for blur detector to fire")
ap.add_argument("-v", "--vis", type=int, default=-1, help="whether or not we are visualizing intermediary steps")
ap.add_argument("-d", "--test", type=int, default=-1, help="whether we should progressively blur the image")

args = vars(ap.parse_args())


orig = cv2.imread(args["image"])
print(orig)

orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# apply blur detector
(mean, blurry) = detect_blur_fft(gray, size=60, thresh=args["thresh"], vis=args["vis"] > 0)

image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
print("[INFO] {}".format(text))

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


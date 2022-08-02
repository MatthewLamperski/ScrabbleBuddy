from imutils.video import VideoStream
import argparse
import time

from ScrabbleBuddy import ScrabbleBuddy

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", type=int, default=0, help="Which camera source (usually either 0 or 1)")
args = vars(ap.parse_args())

print("[INFO] warming up camera...")
vs = VideoStream(src=args["camera"]).start()
time.sleep(2.0)

sb = ScrabbleBuddy(vs)
sb.root.mainloop()
import math
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
import imutils
import cv2

class ScrabbleBuddy:
    def __init__(self, vs):
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.scan = None

        # Initialize root window and image panel
        self.root = tk.Tk()
        self.panel = None
        self.scanpanel = None

        btn = tk.Button(self.root, text="Scan", command=self.scanimage)
        btn.pack(side="bottom", fill="both", expand=1, padx=10, pady=10)

        # Start thread to get video
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoloop, args=())
        self.thread.start()

        # When window closes
        self.root.wm_title('ScrabbleBuddy')
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoloop(self):
        try:
            while not self.stopEvent.is_set():
                # grab the next frame
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=1000)
                cy, cx = self.get_shape_center()

                # draw cube on frame

                # edge size is hypotenuse, a is opposite, b is adjacent
                edge_size = 200
                deg30 = 30 * (math.pi / 180)
                a = math.sin(deg30) * edge_size
                b = math.cos(deg30) * edge_size
                white = (255, 255, 255)
                adjust_center = 20

                # Create cube sides with polylines
                left_face = np.array(
                    [[cx - b, cy - a], [cx, cy - adjust_center], [cx, cy + (edge_size - adjust_center)],
                     [cx - b, cy + edge_size - a]], dtype=np.int32)
                right_face = np.array([[cx, cy - adjust_center], [cx + b, cy - a], [cx + b, cy + edge_size - a],
                                       [cx, cy + (edge_size - adjust_center)]], dtype=np.int32)
                top_face = np.array(
                    [[cx - b, cy - a], [cx, cy - a - (a - adjust_center)], [cx + b, cy - a], [cx, cy - adjust_center]],
                    dtype=np.int32)

                # put faces of cube on frame
                cv2.polylines(self.frame, [left_face], True, white, thickness=3)
                cv2.polylines(self.frame, [right_face], True, white, thickness=3)
                cv2.polylines(self.frame, [top_face], True, white, thickness=3)

                # turn image readble by tk
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                if self.panel is None:
                    self.panel = tk.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError as e:
            print("[INFO] Caught a RuntimeError")

    def scanimage(self):
        # SCAN

        cy, cx = self.get_shape_center()

        edge_size = 200
        deg30 = 30 * (math.pi / 180)
        a = math.sin(deg30) * edge_size
        b = math.cos(deg30) * edge_size
        white = (255, 255, 255)
        adjust_center = 20

        # Create cube sides with polylines
        left_face = np.array(
            [[cx - b, cy - a], [cx, cy - adjust_center], [cx, cy + (edge_size - adjust_center)],
             [cx - b, cy + edge_size - a]], dtype=np.int32)
        right_face = np.array([[cx, cy - adjust_center], [cx + b, cy - a], [cx + b, cy + edge_size - a],
                               [cx, cy + (edge_size - adjust_center)]], dtype=np.int32)
        top_face = np.array(
            [[cx - b, cy - a], [cx, cy - a - (a - adjust_center)], [cx + b, cy - a], [cx, cy - adjust_center]],
            dtype=np.int32)

        # First set destination points for perspective transform
        t_dst = np.array([[0, 100], [100, 0], [200, 100], [100, 200]], dtype=np.float32)
        lr_dst = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)

        # Copy src points to float array
        t_src = top_face.astype('float32')
        l_src = left_face.astype('float32')
        r_src = right_face.astype('float32')

        # Next get perspective transform matrices
        tM = cv2.getPerspectiveTransform(t_src, t_dst)
        lM = cv2.getPerspectiveTransform(l_src, lr_dst)
        rM = cv2.getPerspectiveTransform(r_src, lr_dst)

        t_warp = cv2.warpPerspective(self.frame, tM, (200, 200))

        # These vars are the perspective changes for each side of cube
        l_warp = cv2.warpPerspective(self.frame, lM, (200, 200))
        r_warp = cv2.warpPerspective(self.frame, rM, (200, 200))
        t_mask = self.get_face(t_dst, t_warp)

        # l_warp = automatic_brightness_and_contrast(l_warp)
        # r_warp = automatic_brightness_and_contrast(r_warp)
        # t_mask = automatic_brightness_and_contrast(t_mask)

        # Go to grayscale
        # l_warp = cv2.cvtColor(l_warp, cv2.COLOR_BGR2GRAY)
        # r_warp = cv2.cvtColor(r_warp, cv2.COLOR_BGR2GRAY)
        # t_mask = cv2.cvtColor(t_mask, cv2.COLOR_BGR2GRAY)
        #
        # # Turn back into 3 channels
        # l_warp = cv2.cvtColor(l_warp, cv2.COLOR_GRAY2BGR)
        # r_warp = cv2.cvtColor(r_warp, cv2.COLOR_GRAY2BGR)
        # t_mask = cv2.cvtColor(t_mask, cv2.COLOR_GRAY2BGR)
        #
        # # Run through gaussian blur to smoothen
        # l_warp = cv2.GaussianBlur(l_warp, (5, 5), 0)
        # r_warp = cv2.GaussianBlur(r_warp, (5, 5), 0)
        # t_mask = cv2.GaussianBlur(t_mask, (5, 5), 0)
        #
        # # Sharpen image
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # l_warp = cv2.filter2D(l_warp, -1, kernel)
        # r_warp = cv2.filter2D(r_warp, -1, kernel)
        # t_mask = cv2.filter2D(t_mask, -1, kernel)
        #
        # l_warp = cv2.cvtColor(l_warp, cv2.COLOR_BGR2RGB)
        # r_warp = cv2.cvtColor(r_warp, cv2.COLOR_BGR2RGB)
        # t_mask = cv2.cvtColor(t_mask, cv2.COLOR_BGR2RGB)

        scans = self.build_scanpannel(t_mask, r_warp, l_warp)

        # turn image readble by tk
        image = cv2.cvtColor(scans, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if self.scanpanel is None:
            self.scanpanel = tk.Label(image=image)
            self.scanpanel.image = image
            self.scanpanel.pack(side="left", padx=10, pady=10)

        else:
            self.scanpanel.configure(image=image)
            self.scanpanel.image = image

    def get_shape_center(self):
        (height, width) = self.frame.shape[:2]
        return height / 2, width / 2

    def get_face(self, face, fram):
        face = face.astype('int32')
        rect = cv2.boundingRect(face)
        x, y, w, h = rect
        cropped = fram[y:y + h, x:x + w].copy()
        top_face = face - face.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [top_face], -1, (255, 255, 255), -1, cv2.LINE_AA)

        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        return dst

    def build_scanpannel(self, t, r, l):
        th, tw = t.shape[:2]
        lh, lw = l.shape[:2]
        rh, rw = r.shape[:2]

        card = np.zeros((max(th, lh, rh), tw + lw + rw, 3), dtype="uint8")
        card[0:th, 0:tw] = t
        card[0:lh, tw:tw + lw] = l
        card[0:rh, tw + lw:tw + lw + rw] = r

        return card


    def onClose(self):
        # set stop event, cleanup camera, allow rest of quit process to continue
        print("[INFO] Closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()

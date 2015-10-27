import os
import glob
import time
import timeit
import dlib
from skimage import io


def train():
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = True
    options.C = 5
    options.num_threads = 4
    options.be_verbose = True
    options.detection_window_size = 1024
    options.match_eps = 0.1

    training_xml_path = "../pics/train/training-single.xml"
    # training_xml_path = "/home/external/moderation-porn-detector/oboobs.dlibxml"

    dlib.train_simple_object_detector(training_xml_path, "../boobs.svm", options)

    print("Training accuracy: {}".format(dlib.test_simple_object_detector(training_xml_path, "../boobs.svm")))


def test():
    detector = dlib.simple_object_detector("../boobs.svm")

    win_det = dlib.image_window()
    win_det.set_image(detector)
    # dlib.hit_enter_to_continue()

    print("Showing detections on the images in the faces folder...")
    win = dlib.image_window()

    tp = 0
    fn = 0
    dur = 0

    test_path = "../pics/test/*.jpg"
    # test_path = "/home/external/moderation-porn-detector/boobs-oboobs/*.jpg"

    for f in glob.glob(test_path):
        try:
            img = io.imread(f)
        except IOError as e:
            print("Image {} can't be loaded: {}".format(f, e.message))
            continue

        try:
            t_start = time.clock()
            dets = detector(img)
            t_end = time.clock()
        except RuntimeError as e:
            print("Image {} can't be detected: {}".format(f, e.message))
            continue

        dur += (t_end - t_start)

        num = len(dets)
        if num > 0:
            print("Boobs {} detected in file: {}".format(num, f))
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))

            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(dets)
            # dlib.hit_enter_to_continue()

            tp += 1
        else:
            fn += 1

        if tp + fn > 100:
            break

    print("tp={} fn={} precision={} dur={}".format(tp, fn, 1.0 * tp / (tp + fn), 1.0 * dur / (tp + fn)))


def camera_boobs():
    import cv2

    # cam = cv2.VideoCapture(-1)
    cam = cv2.VideoCapture("/home/nick/temp/boobs/b1.flv")

    win = dlib.image_window()
    detector = dlib.simple_object_detector("detector.svm")

    def detect():
        s, img = cam.read()
        if not s:
            return

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dets = detector(img2)
        num = len(dets)

        if num > 0:
            print("found!")

        win.clear_overlay()
        win.set_image(img2)
        win.add_overlay(dets)

    try:
        while True:
            detect()
    except KeyboardInterrupt as e:
        print("exiting")

    cam.release()
    cv2.destroyAllWindows()


def camera_fd():
    import cv2

    cam = cv2.VideoCapture(-1)

    win = dlib.image_window()
    detector = detector = dlib.get_frontal_face_detector()

    def detect():
        s, img = cam.read()
        if not s:
            return

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dets = detector(img2)
        num = len(dets)

        if num > 0:
            print("found!")

        win.clear_overlay()
        win.set_image(img2)
        win.add_overlay(dets)

    try:
        while True:
            detect()
    except KeyboardInterrupt as e:
        print("exiting")

    cam.release()
    cv2.destroyAllWindows()

# train()
test()
# camera_fd()
# camera_boobs()

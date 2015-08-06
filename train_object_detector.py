import os
import glob
import time
import timeit
import dlib
from skimage import io

faces_folder = "pics"


def train():
    # Now let's do the training.  The train_simple_object_detector() function has a
    # bunch of options, all of which come with reasonable default values.  The next
    # few lines goes over some of these options.
    options = dlib.simple_object_detector_training_options()
    # Since faces are left/right symmetric we can tell the trainer to train a
    # symmetric detector.  This helps it get the most value out of the training
    # data.
    options.add_left_right_image_flips = True
    # The trainer is a kind of support vector machine and therefore has the usual
    # SVM C parameter.  In general, a bigger C encourages it to fit the training
    # data better but might lead to overfitting.  You must find the best C value
    # empirically by checking how well the trained detector works on a test set of
    # images you haven't trained on.  Don't just leave the value set at 5.  Try a
    # few different C values and see what works best for your data.
    options.C = 5
    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 4
    options.be_verbose = True

    training_xml_path = os.path.join(faces_folder, "train", "training.xml")

    # This function does the actual training.  It will save the final detector to
    # detector.svm.  The input is an XML file that lists the images in the training
    # dataset and also contains the positions of the face boxes.  To create your
    # own XML files you can use the imglab tool which can be found in the
    # tools/imglab folder.  It is a simple graphical tool for labeling objects in
    # images with boxes.  To see how to use it read the tools/imglab/README.txt
    # file.  But for this example, we just use the training.xml file included with
    # dlib.
    dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

    # Now that we have a face detector we can test it.  The first statement tests
    # it on the training data.  It will print(the precision, recall, and then)
    # average precision.
    print("")  # Print blank line to create gap from previous output
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, "detector.svm")))


def test():
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector("detector.svm")
    # detector = dlib.get_frontal_face_detector()

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!
    # win_det = dlib.image_window()
    # win_det.set_image(detector)
    # dlib.hit_enter_to_continue()

    # Now let's run the detector over the images in the faces folder and display the
    # results.
    print("Showing detections on the images in the faces folder...")
    win = dlib.image_window()

    tp = 0
    fn = 0
    dur = 0

    # test_path = os.path.join(faces_folder, "test", "*.jpg")
    test_path = "/home/external/moderation-porn-detector/boobs-oboobs/*.jpg"

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

train()
test()

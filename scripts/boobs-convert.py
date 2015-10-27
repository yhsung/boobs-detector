import glob
import os
import xml.dom.minidom as minidom

dlib_xml = minidom.Document()

dlib_dataset = dlib_xml.createElement("dataset")
dlib_xml.appendChild(dlib_dataset)

dlib_name = dlib_xml.createElement("name")
dlib_name.appendChild(dlib_xml.createTextNode("Generated training set"))
dlib_dataset.appendChild(dlib_name)

dlib_images = dlib_xml.createElement("images")
dlib_dataset.appendChild(dlib_images)

opencv_file = open("/home/external/moderation-porn-detector/oboobs.opencvdat", "w")

for srcfile in glob.glob("/home/external/moderation-porn-detector/boobs-oboobs/*.xml"):
    srcdoc = minidom.parse(srcfile)

    src_rects = srcdoc.getElementsByTagName("rects")[0].getElementsByTagName("rect")
    src_rects_len = len(src_rects)
    if src_rects_len == 0:
        continue

    src_annotation = srcdoc.getElementsByTagName("annotation")[0]
    filename = os.path.join(os.path.dirname(srcfile), src_annotation.getAttribute("filename"))

    dlib_image = dlib_xml.createElement("image")
    dlib_image.setAttribute("file", filename)
    dlib_images.appendChild(dlib_image)

    opencv_file.write("{}  {}".format(filename, src_rects_len))

    for src_rect in src_rects:
        x = src_rect.getAttribute("x")
        y = src_rect.getAttribute("y")
        w = src_rect.getAttribute("w")
        h = src_rect.getAttribute("h")

        area = float(w) * float(h)
        ratio = float(w) / float(h)
        if area >= 400 and 0.85 < ratio < 1.15:
            dlib_box = dlib_xml.createElement("box")
            dlib_box.setAttribute("left", x)
            dlib_box.setAttribute("top", y)
            dlib_box.setAttribute("width", w)
            dlib_box.setAttribute("height", h)
            dlib_image.appendChild(dlib_box)

        opencv_file.write("  {} {} {} {}".format(x, y, w, h))

    opencv_file.write("\n")

opencv_file.close()

f = open("/home/external/moderation-porn-detector/oboobs.dlibxml", 'w')
f.write(dlib_xml.toprettyxml(indent=" ", encoding="UTF-8"))
f.close()

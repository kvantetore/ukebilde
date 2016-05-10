# -*- coding: utf-8 -*-
import cv2
import numpy
import os
import math


def get_target_path(filename):
    path, name = os.path.split(filename)
    target_folder = "./scaled"
    return os.path.join(target_folder, name)


def resize_face(filename, target_path, source_pos, target_pos):
    position_filename = os.path.splitext(filename)[0] + ".txt"

    #determine if file or target file has changed
    should_update = False
    if not os.path.exists(target_path):
        #print "update if we don't have a target file"
        should_update = True
    elif os.path.getmtime(target_path) < os.path.getmtime(filename):
        #print "update if source file is newer than target file"
        should_update = True
    elif os.path.exists(position_filename) and os.path.getmtime(target_path) < os.path.getmtime(position_filename):
        #print "update if position file is newer than target file"
        should_update = True

    if not should_update:
        return

    #load source mage
    orig = cv2.imread(filename)
    dst = numpy.array(orig)

    source_leye = numpy.array(source_pos[0, :])
    source_reye = numpy.array(source_pos[1, :])
    source_nose = numpy.array(source_pos[2, :])

    target_leye = numpy.array(target_pos[0, :])
    target_reye = numpy.array(target_pos[1, :])
    target_nose = numpy.array(target_pos[2, :])

    #vector from left to right eye
    source_ltor_eye = source_reye - source_leye
    target_ltor_eye = target_reye - target_leye

    ##scale & rotate
    scale = math.sqrt(numpy.dot(target_ltor_eye, target_ltor_eye)) / math.sqrt(
        numpy.dot(source_ltor_eye, source_ltor_eye))
    angle = 360 * math.atan2(source_ltor_eye[1], source_ltor_eye[0]) / (2 * math.pi)
    transform = cv2.getRotationMatrix2D(tuple(source_leye), angle, scale)
    transform[:, 2] -= source_leye - target_leye

    #transform image
    dst = cv2.warpAffine(orig, transform, (orig.shape[1], orig.shape[0]), dst, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT,
                         (200, 200, 200))

    #crop
    height = 1600
    width = height * 16 / 9
    p1x, p1y = target_nose[1] - height / 2 - 100, target_nose[0] - width / 2
    p2x, p2y = target_nose[1] + height / 2 - 100, target_nose[0] + width / 2
    dst = dst[p1x:p2x, p1y:p2y]

    #add vignetting
    center = dst.shape[1] / 2, dst.shape[0] / 2
    dst = vignetting(dst, center, 500, 400)

    #save transformed image
    target_folder = os.path.split(target_path)[0]
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    cv2.imwrite(target_path, dst)


def vignetting(image, center, radius, feather):
    target_color = numpy.array([40, 40, 40])

    x, y, c = numpy.ix_(range(image.shape[0]), range(image.shape[1]), range(image.shape[2]))

    dist = numpy.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    amount = numpy.minimum(1, numpy.maximum(0, (dist - radius) / feather))
    image = numpy.array(target_color * amount + image * (1 - amount), dtype=numpy.uint8)

    return image

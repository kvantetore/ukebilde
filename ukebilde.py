# -*- coding: utf-8 -*-
import numpy
import os
import json
import argparse

from find_face import find_face_position_manual, save_face_position, find_stored_face_position
from resize_face import resize_face

TARGET_POS = numpy.array([
    #left eye
    [
        1608.0,
        1152.0
    ],
    #right eye
    [
        1983.0,
        1197.0
    ],
    #nose
    [
        1794.0,
        1308.0
    ]
], dtype=numpy.float32)


FIND_FACE = "find_face"
RESIZE_FACE = "resize_face"
available_actions = [FIND_FACE, RESIZE_FACE]


def find_files():
    return sorted()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", action="append", choices=available_actions, default=[])
    parser.add_argument("files", nargs="*", default=[])
    parser.add_argument("--folder", default=".")
    parser.add_argument("--force", "-f", action="store_true")

    args = parser.parse_args()

    #get list of files
    files = []
    if not args.files:
        folder = os.path.abspath(args.folder)
        file_names = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        files = [os.path.join(folder, f) for f in file_names]
    else:
        files = [os.path.abspath(f) for f in args.files if f.endswith(".jpg")]

    if not args.action:
        args.action = available_actions

    if FIND_FACE in args.action:
        for filename in files:
            pos = find_stored_face_position(filename)
            if pos is None or args.force:
                pos = find_face_position_manual(filename)
                save_face_position(filename, pos)

    if RESIZE_FACE in args.action:
        for filename in files:
            print "Resizing %s" % filename
            pos = find_stored_face_position(filename)
            if pos is None:
                raise Exception("Missing face position file for " + filename)

            resize_face(filename, pos, TARGET_POS)


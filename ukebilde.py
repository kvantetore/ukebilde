#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import os
import argparse
import json

from find_face import find_face_position_manual, find_face_position_automatic, save_face_position, find_stored_face_position
from resize_face import resize_face, get_target_path
from create_video import render_frames

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
RENDER_FRAMES = "render_frames"
CREATE_VIDEO = "create_video"
available_actions = [FIND_FACE, RESIZE_FACE, RENDER_FRAMES, CREATE_VIDEO]


def find_files():
    return sorted()

import exifread
import datetime

def get_capture_date(filename):
    with open(filename, "rb") as f:
        tags = exifread.process_file(f, details=False)

    exif_date = tags["EXIF DateTimeOriginal"]
    return datetime.datetime.strptime(exif_date.values, "%Y:%m:%d %H:%M:%S")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", action="append", choices=available_actions, default=[])
    parser.add_argument("files", nargs="*", default=[])
    parser.add_argument("--folder", default="./in")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--settings", "-s", default="settings.json")

    args = parser.parse_args()

    #parse json
    with open(args.settings, "r") as f:
        settings = json.load(f)

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
                pos = find_face_position_automatic(filename)
                save_face_position(filename, pos)

    if RESIZE_FACE in args.action:
        for filename in files:
            print("Resizing %s" % filename)
            pos = find_stored_face_position(filename)
            if pos is None:
                raise Exception("Missing face position file for " + filename)

            target_path = get_target_path(filename)
            resize_face(filename, target_path, pos, TARGET_POS)

    if RENDER_FRAMES in args.action:
        target_paths = [get_target_path(f) for f in files]
        capture_dates = [get_capture_date(f) for f in files]
        render_frames(target_paths, capture_dates, settings)

    #if CREATE_VIDEO in args.action:

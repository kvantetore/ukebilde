from __future__ import division
import cv2
import os
from collections import namedtuple
import exifread
from functools32 import lru_cache

out_folder = "./frames"

FramePart = namedtuple("FramePart", "filename capture_date weight")
Frame = namedtuple("Frame", "frame_index frame_parts")

@lru_cache(maxsize=4)
def read_image(filename, video_size):
    print "reading", filename
    img = cv2.imread(filename)
    return cv2.resize(img, video_size)


class Timeline():
    def __init__(self, filenames, capture_dates, delay, fade):
        self.fps = 24
        self.video_size = (1280, 720)
        self.filenames = filenames
        self.capture_dates = capture_dates
        self.delay = delay
        self.fade = fade
        self.frames = self.create_frames()

        self.start_date = capture_dates[0]

    def create_frames(self):
        blend_frames = int(self.fade * self.fps)
        delay_frames = int(self.delay * self.fps)

        frames = []
        file_index = 0
        while file_index < len(self.filenames):
            #blend from previous file
            if file_index > 0:
                for i in xrange(blend_frames):
                    weight = i / blend_frames
                    frame = Frame(len(frames), [
                        FramePart(self.filenames[file_index-1], self.capture_dates[file_index-1], 1 - weight),
                        FramePart(self.filenames[file_index], self.capture_dates[file_index], weight),
                    ])
                    frames.append(frame)

            for i in xrange(delay_frames):
                frame = Frame(len(frames), [
                    FramePart(self.filenames[file_index], self.capture_dates[file_index], 1.0),
                ])
                frames.append(frame)

            file_index += 1

        return frames

    def read_image(self, filename):
        return read_image(filename, self.video_size)

    def render_frame(self, frame):
        parts = frame.frame_parts


        #blend between frame parts
        source = self.read_image(parts[0].filename)
        if len(parts) == 1:
            pass
        elif len(parts) == 2:
            source1 = self.read_image(parts[1].filename)
            source = cv2.addWeighted(source, parts[0].weight, source1, parts[1].weight, -1)
        else:
            raise Exception("Invalid number of frame parts for frame ", frame)

        age = parts[0].capture_date - self.capture_dates[0]
        months = int(age.days / 30.4)

        if months > 0:
            cv2.putText(source, "%imnd" % months, (int(source.shape[1]/20), int(source.shape[1]/15)), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 5)

        out_path = os.path.join(out_folder, "frame-{:0>5d}.jpg".format(frame.frame_index))
        print "writing", out_path

        cv2.imwrite(out_path, source)

    def render(self):
        for frame in self.frames:
            self.render_frame(frame)


def render_frames(filenames, capture_dates, delay=.3, fade=0.2):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    timeline = Timeline(filenames, capture_dates, delay, fade)
    timeline.render()

def create_video():
    pass

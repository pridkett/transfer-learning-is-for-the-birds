#!/usr/bin/env python
"""
flask_streamer.py

Based off https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited
"""
import argparse

from PIL import Image
from PIL import ImageDraw

import os
from flask import Flask, render_template, Response

from camera_pi import Camera

import detect
import detect_image as di
import time

from io import BytesIO

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()

        image_file = BytesIO(frame)
        image = Image.open(image_file)
        scale = detect.set_input(interpreter, image.size,
                                 lambda size: image.resize(size, Image.ANTIALIAS))

        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_output(interpreter, threshold, scale)
        objs = [x for x in objs if x.score > threshold]
        print('%.2f ms' % (inference_time * 1000))

        print('-------RESULTS--------')
        if not objs:
            print('No objects detected')

        for obj in objs:
            print(labels.get(obj.id, obj.id))
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)

        di.draw_objects(ImageDraw.Draw(image), objs, labels)

        outbytes = BytesIO()
        image.save(outbytes, format="JPEG")
        outbytes = outbytes.getvalue()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + outbytes + b'\r\n')
        print('frame rendered')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def init():
    global interpreter
    global labels
    global args
    global threshold
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-l', '--labels', help='File path of labels file.')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.6,
        help='Classification score threshold')
    args = parser.parse_args()

    labels = di.load_labels(args.labels) if args.labels else {}
    threshold = args.threshold

    interpreter = di.make_interpreter(args.model)
    interpreter.allocate_tensors()


if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', threaded=True)

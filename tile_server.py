import time

import skimage
from flask import Flask, abort, send_file
from objects.buildings import BuildingConfig
import os, platform
from mrcnn import model as modellib, visualize
from PIL import Image, ImageColor
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

PLATFORM = platform.platform()
print(PLATFORM)
ROOT_DIR = os.path.abspath("/home/ubuntu/thomas/")

if (PLATFORM.startswith("Darwin")):
    ROOT_DIR = os.path.abspath("/Users/tingold/code/thomas/")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS = os.path.join(ROOT_DIR, 'buildings_latest.h5')


class InferenceConfig(BuildingConfig):
    IMAGES_PER_GPU = 1


config = InferenceConfig()

global model
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
model.load_weights(WEIGHTS, by_name=True)
global graph
graph = tf.get_default_graph()


@app.route('/', methods=['GET'])
def index():
    return send_file('map.html', "text/html")


@app.route('/tiles/<int:z>/<int:x>/<int:y>', methods=['GET'])
def tile(z, x, y):
    request_start_time = time.time()
    with graph.as_default():
    # if z != 18:
    # only service level 18 for now
    #    abort(404)
        url = "https://b.tiles.mapbox.com/v4/mapbox.satellite/{}/{}/{}.png?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NDg1bDA1cjYzM280NHJ5NzlvNDMifQ.d6e-nNyBDtmQCVwVNivz7A".format(z, x, y)
        image = skimage.io.imread(url)
        r = model.detect([image], verbose=1)[0]
        # output = Image.new('RGBA',(256,256),(0,0,0,0))
        for i in range(len(r['rois'])):
            if r['scores'][i] > 0.93:
                image = visualize.draw_box(image, r['rois'][i], (255, 0, 0))
                image = visualize.apply_mask(image, r['masks'][i], (255, 0, 0))
        output = Image.fromarray(image, 'RGB')
        byte_io = BytesIO()
        output.save(byte_io, 'PNG')
        byte_io.seek(0)
        request_end_time = time.time();
        return send_file(byte_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)

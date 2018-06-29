"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
from datetime import datetime

import numpy as np
import skimage
from PIL import Image
import platform
from mrcnn import model as modellib, visualize

# Root directory of the project
PLATFORM = platform.platform()
print(PLATFORM + "(Model)")
ROOT_DIR = os.path.abspath("/home/ubuntu/thomas/")

if(PLATFORM.startswith("Darwin")):
    ROOT_DIR = os.path.abspath("/Users/tingold/code/thomas/")

MODEL_DIR = os.path.join(ROOT_DIR,"logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class BuildingConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "buildings"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    USE_MINI_MASK = False



class BuildingDataset(utils.Dataset):

    #PATH = '/Users/tingold/code/Mask_RCNN/samples/objects/training_data'
    PATH = os.path.join(ROOT_DIR,'samples/objects/training_data')

    image_lookup = []

    def load_buildings(self,):

        self.add_class("objects", 1, "building")
        print("Loading objects")

        image_filenames = os.listdir(self.PATH + '/sat')
        cnt = 0
        for img_file in image_filenames:
            # id is the tile name without sat in front
            id = img_file.replace("sat","", 1)

            abs_img = self.PATH + "/sat/" + img_file
            self.image_lookup.insert(cnt, id)
            self.add_image("objects", image_id=cnt, path=abs_img, width=256, height=256)

    def load_mask(self, image_id):
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.

        # print("Loading mask for image id "+self.image_lookup[image_id])


        mask_url = self.PATH+'/osm/osm'+self.image_lookup[image_id]
        # Pack instance masks into an array
        mask = Image.open(mask_url)
        red, green, blue, alpha = mask.split()
        #mask = skimage.io.imread("file://"+mask_url, as_gray=False)
        data = np.array(red)

        # hot = lambda v: 0 if v < 1 else 1
        #f = np.vectorize(hot)
        # bin_data = f(data)
        class_ids = np.array([1])
        class_ids[0] = 1

        return data, class_ids


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        # print("Loading image for image id " + self.image_lookup[image_id])
        img_url = self.PATH + '/sat/sat' + self.image_lookup[image_id]
        # Pack instance masks into an array
        img = Image.open(img_url)
        return np.array(img)

def detect(model):
    print("Running on {}".format(args.img))
    # Read image
    image = skimage.io.imread(args.img)
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    for i in range(len(r['rois'])):
        image = visualize.draw_box(image,r['rois'][i],(255,0,0))
        image = visualize.apply_mask(image,r['masks'][i],(255,0,0))

    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.now())
    skimage.io.imsave(file_name, image)

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'run'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file ")
    parser.add_argument("--img", required=False, metavar="image.jpg", help="an image to analyze")
    args = parser.parse_args()

    if args.command == "train":

        config = BuildingConfig()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
        # Training
        dataset_train = BuildingDataset()
        dataset_train.load_buildings()
        dataset_train.prepare()
        # Validation
        dataset_val = BuildingDataset()
        dataset_val.load_buildings()
        dataset_val.prepare()

        model.load_weights(args.weights, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

        model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')


    if args.command == "run":

        class InferenceConfig(BuildingConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)
        model.load_weights(args.weights, by_name=True)
        detect(model)
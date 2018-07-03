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
import glob
from mrcnn import model as modellib, visualize

# Root directory of the project
PLATFORM = platform.platform()
print(PLATFORM + "(Model)")
ROOT_DIR = os.path.abspath("/home/ubuntu/thomas/")

if (PLATFORM.startswith("Darwin")):
    ROOT_DIR = os.path.abspath("/Users/tingold/code/thomas/")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
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
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    USE_MINI_MASK = False


class BuildingDataset(utils.Dataset):
    # PATH = '/Users/tingold/code/Mask_RCNN/samples/objects/training_data'
    PATH = os.path.join(ROOT_DIR, 'objects/training_data')

    image_lookup = []

    def load_buildings(self, ):
        self.add_class("buildings", 1, "building")
        print("Loading buildings")

        image_filenames = os.listdir(self.PATH + '/sat')
        cnt = 0
        for img_file in image_filenames:
            # id is the tile name without sat in front
            id = img_file.replace("sat", "", 1)

            abs_img = self.PATH + "/sat/" + img_file
            self.image_lookup.insert(cnt, id)
            self.add_image("buildings", image_id=cnt, path=abs_img, width=256, height=256)

    def load_mask(self, image_id):
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.

        # print("Loading mask for image id "+self.image_lookup[image_id])

        real_image_id = self.image_lookup[image_id]
        mask_dir = self.PATH + '/osm/' + real_image_id + '/**'
        masks = glob.glob(mask_dir)
        # masks should be an array of file names like building-0-Z_X_Y.png
        mask_array = np.empty((256, 256, len(masks)), dtype=np.uint8)

        for i, m in enumerate(masks):
            mask_image = Image.open(m)
            red, green, blue, alpha = mask_image.split()
            # Pack masks into an array
            mask_array[..., i - 1] = np.asarray(red)

        class_ids = np.ones([len(masks)])
        return mask_array, class_ids

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
        image = visualize.draw_box(image, r['rois'][i], (255, 0, 0))
        image = visualize.apply_mask(image, r['masks'][i], (255, 0, 0))

    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.now())
    skimage.io.imsave(file_name, image)


def mask_tile(image,masks, color, alpha=0.5):
    """ Return and RGBA image that is transparent except for the mask
    """
    mask = (np.sum(masks, -1, keepdims=True, dtype=int) >= 1) * 1
    # image = np.zeros((256, 256, 4))
    # image[:,:,0] = 256
    # image[:,:,3] = 256
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask[:,:,0] == 1,
                                  image[:, :, c] * color[c],
                                  #(1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


# transparent image
#
# image[:, :, 0] = 255
# image[:, :, 3] = 255
#
# if masks.shape[-1] > 0:
#     mask = (np.sum(masks, -1, keepdims=True, dtype=int) >= 1) * 255
#     image = visualize.apply_mask(image,mask[:,:,0],(255,0,0))

# image.paste(mask,(0,0),mask=mask)
# img = np.zeros((256, 256, 4))
# img[:, :, 0] = mask[:, :, 0]
# img[:, :, 3] = mask[:, :, 0]
# im
# image = Image.fromarray(img, 'RGBA')
# image.put

# a = Image.fromarray(np.full((256,256),255),'L')
# r = Image.fromarray(mask, 'L')
# image = Image.merge('RGBA', (r, g, b, a))
# image.paste(image,(0,0),mask_image)
# return image

# gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
# # Copy color pixels from the original color image where mask is set
# if mask.shape[-1] > 0:
#     # We're treating all instances as one, so collapse the mask into one layer
#     mask = (np.sum(mask, -1, keepdims=True) >= 1)
#     splash = np.where(mask, image, gray).astype(np.uint8)
# else:
#     splash = gray.astype(np.uint8)
# return splash


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

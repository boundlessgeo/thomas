source activate tensorflow_p36
export PYTHONPATH=$PYTHONPATH:.
pip install -r requirements.txt
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
python samples/buildings/buildings.py &
tensorboard --logdir logs &
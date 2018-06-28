python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

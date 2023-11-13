# copy edited https://colab.research.google.com/drive/1QtTW9-ukX2HKZGvt0QvVGqjuqEykoZKI
# Install detectron2
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch

# Use the below line to install detectron2 if the above one has an error
# sudo apt-get install build-essential
# sudo apt-get install python3.10-dev
# pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
# cd Detic
# pip install -r requirements.txt

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(1, 'Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test

def check_torch():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Build the detector and download our pretrained weights
def build_cfg(device: str):
    """Build the detector and download our pretrained weights

    Args:
        device (str): 'cpu' use cpu-only mode, default use gpu

    Returns:
        CfgNode: config
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    if device=='cpu':
        cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
    return cfg
    
def build_model(cfg: CfgNode):
    predictor = DefaultPredictor(cfg)
    # # Setup the model's vocabulary using build-in datasets
    # BUILDIN_CLASSIFIER = {
    #     'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    #     'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    #     'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    #     'coco': 'datasets/metadata/coco_clip_a+cname.npy',
    # }

    # BUILDIN_METADATA_PATH = {
    #     'lvis': 'lvis_v1_val',
    #     'objects365': 'objects365_v2_val',
    #     'openimages': 'oid_val_expanded',
    #     'coco': 'coco_2017_val',
    # }
    # vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
    # metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
    # classifier = BUILDIN_CLASSIFIER[vocabulary]
    # num_classes = len(metadata.thing_classes)
    # reset_cls_test(predictor.model, classifier, num_classes)
    return predictor

# Change the model's vocabulary to a customized one and get their word-embedding 
#  using a pre-trained CLIP model.
from Detic.detic.modeling.text.text_encoder import build_text_encoder
def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def create_detector(device: str = 'cpu'):
    cfg = build_cfg(device)
    print("Setup the detector use {}.".format(device))
    return build_model(cfg)

def detect_object(predictor: DefaultPredictor,img_path: str, threshold: float = 0.5, classes: list = ['headphone', 'webcam', 'paper', 'coffee'], view: bool = False):
    im = cv2.imread(img_path)
    vocabulary = 'custom'
    metadata = MetadataCatalog.get("__unused")
    metadata.thing_classes = classes # Change here to try your own vocabularies!
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    # Reset visualization threshold
    output_score_threshold = threshold
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

    # Run model and show results
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im_cv = out.get_image()[:, :, ::-1]
    if view:
        cv2.imshow("Show result", im_cv)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # delete current classes for next detect
    del metadata.thing_classes
    return im_cv, len(outputs["instances"])

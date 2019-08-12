import datetime
import time

import torch
import logging

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug

from maskrcnn_benchmark.data.transforms import transforms as T
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from inference.configuration import Configuration
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model


from threading import Lock
from torch.utils.data import DataLoader
from .memory_files import MemoryFiles
import pycocotools.mask as mask_util
import numpy as np

class InferenceService(object):
    def __init__(self, conf: Configuration, logger=logging.getLogger("inference_service")):
        self.conf = conf
        self.logger = logger

        cfg.merge_from_file(conf.config_file)
        cfg.freeze()
        self.cfg = cfg

        self.model = None
        self.device = cfg.MODEL.DEVICE
        self.process_lock = Lock()

        self.transforms = None

        self._start()
    def _start(self):
        # Load neural network
        self.model = build_detection_model(cfg)
        self.model.to(cfg.MODEL.DEVICE)

        self.logger.info("Model loaded")
        output_dir = self.cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.cfg, self.model, save_dir=output_dir)
        _ = checkpointer.load(self.conf.weight_file)
        self.logger.info("Weight loaded")

        self.model.eval()

        self.transforms = None if self.cfg.TEST.BBOX_AUG.ENABLED else self.build_inference_transform()

    # Image shape must be the same within one batch
    def process(self, images, image_shape):
        with self.process_lock:
            start_time = time.time()
            dataloader = DataLoader(
                MemoryFiles(images, self.transforms, self.cfg.DATALOADER.SIZE_DIVISIBILITY),
                batch_size=self.conf.batch_size,
                shuffle=False,
                num_workers=self.conf.n_cpu,
            )
            cpu_device = torch.device("cpu")
            detections_list = []
            for batch_i, (input_imgs) in enumerate(dataloader):
                with torch.no_grad():
                    if self.cfg.TEST.BBOX_AUG.ENABLED:
                        output = im_detect_bbox_aug(self.model, input_imgs, self.device)
                    else:
                        output = self.model(input_imgs.to(self.device))

                    if len(output):
                        output = [o.to(cpu_device) for o in output]
                        detections_list.extend(output)
            inference_time = datetime.timedelta(seconds=time.time() - start_time)
            self.logger.info(f"Inferred batch (n={len(images)}). Inference time={inference_time}. Time per frame={inference_time/len(images)}")
        # CPU task can be executed in parallel
        return self.extract_information(detections_list, image_shape)

    def extract_information(self, detections_in_all_images, image_shape):
        cropped_threshold = self.conf.cropped_threshold
        result = []
        masker = Masker(threshold=0.5, padding=1)

        for img_index, detections in enumerate(detections_in_all_images):
            detections.resize(image_shape)
            masks = detections.get_field("mask")
            masks = masker(masks.expand(1, -1, -1, -1, -1), detections)
            masks = masks[0]

            num_detections = len(detections)
            mode = detections.mode
            bbox = detections.bbox.tolist()
            scores = detections.extra_fields["scores"].tolist()
            labels = detections.extra_fields["labels"].tolist()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            for i in range(num_detections):
                result.append({
                    "img": img_index,
                    "bboxes": bbox[i],
                    "scores":scores[i],
                    "labels": labels[i],
                    #"masks": masks[i],
                    "rles":rles,
                    "mode": mode,
                    "is_cropped": None}
                )
        return result

    def build_inference_transform(self):
        # discard the unused transforms for efficiency
        cfg = self.cfg
        to_bgr255 = cfg.INPUT.TO_BGR255
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )
        return [T.Resize(min_size, max_size), T.ToTensor(), normalize_transform]




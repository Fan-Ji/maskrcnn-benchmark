import datetime
import time

import torch
import logging

from inference.memory_files import MemoryFilesCollator
from inference.memory_files import MemoryFiles
from inference.configuration import Configuration

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.data.transforms import transforms as T
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model

from threading import Lock
from torch.utils.data import DataLoader
import pycocotools.mask as mask_util
import numpy as np


class InferenceService(object):
    def __init__(self, conf: Configuration, logger=logging.getLogger("inference_service")):
        self.conf = conf
        self.logger = logger

        self.cfg = self.load_cfg(self.conf)

        self.model = None
        self.device = cfg.MODEL.DEVICE
        self.process_lock = Lock()

        self.transforms = None

        self._start()

    @staticmethod
    def load_cfg(conf):
        config = cfg.clone()
        config.merge_from_file(conf.config_file)
        config.freeze()
        return config

    def _start(self):
        # Load neural network
        self.model = build_detection_model(self.cfg)
        self.model.to(self.cfg.MODEL.DEVICE)

        self.logger.info("Model loaded")
        output_dir = self.cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.cfg, self.model, save_dir=output_dir)
        _ = checkpointer.load(self.conf.weight_file)
        self.logger.info("Weight loaded")

        self.model.eval()

        self.transforms = None if self.cfg.TEST.BBOX_AUG.ENABLED else self.build_inference_transform(self.cfg)

    # Image shape must be the same within one batch
    def process(self, images):
        with self.process_lock:
            start_time = time.time()
            dataloader = DataLoader(
                MemoryFiles(images, self.transforms),
                batch_size=self.conf.batch_size,
                shuffle=False,
                num_workers=self.conf.n_cpu,
                collate_fn=MemoryFilesCollator(self.cfg.DATALOADER.SIZE_DIVISIBILITY)
            )
            cpu_device = torch.device("cpu")
            detections_list = []
            for batch_i, (input_imgs) in enumerate(dataloader):
                detections_list.extend(self.run_inference(input_imgs))
            inference_time = datetime.timedelta(seconds=time.time() - start_time)
            self.logger.info(
                f"Inferred batch (n={len(images)}). Inference time={inference_time}. Time per frame={inference_time / len(images)}")
        # CPU task can be executed in parallel
        return detections_list

    def run_inference(self, imgs_minibatch):
        with torch.no_grad():
            if self.cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(self.model, imgs_minibatch, self.device)
            else:
                output = self.model(imgs_minibatch.to(self.device))

            if len(output):
                output = [o.to("cpu") for o in output]
            return output

    @staticmethod
    def extract_information_one(detections, image_shape, original_filenames):
        masker = Masker(threshold=0.5, padding=1)
        # The resize does not handle crop: the padding area was still there so the bounding box shrink to the left!?
        # It seems like the original implementation stored un-padded size!
        # Problem solved. See the wiki https://github.com/nncrystals/maskrcnn-benchmark/wiki/Inference-procedures
        detections = detections.resize(image_shape)
        masks = detections.get_field("mask")
        # Caution: if the mask offsets, it is because the bounding box!
        masks = masker(masks.expand(1, -1, -1, -1, -1), detections)
        masks = masks[0]

        num_detections = len(detections)
        bbox = detections.bbox.tolist()
        scores = detections.extra_fields["scores"].tolist()
        labels = detections.extra_fields["labels"].tolist()
        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        results = []
        for i in range(num_detections):
            results.append({
                "name": original_filenames,
                "bbox": bbox[i],
                "score": scores[i],
                "label": labels[i],
                "rle": rles[i],
                "is_cropped": None}
            )
        return results

    def extract_information(self, detections_in_all_images, image_shape, original_filenames=None, options=None):
        result = []

        for img_index, detections in enumerate(detections_in_all_images):
            result.append(self.extract_information_one(detections, image_shape, img_index, original_filenames, options))
        return result

    @staticmethod
    def build_inference_transform(cfg):
        # discard the unused transforms for efficiency
        to_bgr255 = cfg.INPUT.TO_BGR255
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )
        return [T.Resize(min_size, max_size), T.ToTensor(), normalize_transform]

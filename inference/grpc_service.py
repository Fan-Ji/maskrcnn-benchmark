import queue
import time

import grpc
from concurrent import futures

import sys
import logging
import os
import threading
import io
import multiprocessing
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from inference.memory_files import MemoryFiles, MemoryFilesCollator

import inference.proto.inference_service_pb2_grpc as grpc_service
import inference.proto.inference_service_pb2 as grpc_def
from inference.configuration import cfg
from inference.service import InferenceService
import torch.multiprocessing as mp

# logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InferenceServiceImpl")
logger.setLevel(logging.INFO)


class InferenceServiceImpl(grpc_service.InferenceServicer):
    def __init__(self):
        super(InferenceServiceImpl, self).__init__()
        self.inference_batch_max_size = 8
        self.raw_image_queue = mp.Queue()
        self.inference_process = mp.Process(target=self.inference_task,
                                            args=(self.raw_image_queue, self.inference_batch_max_size))
        self.transforms = InferenceService.build_inference_transform(InferenceService.load_cfg(cfg))
        self.logger = logging.getLogger("GRPCInferenceGenerator")

        self.inference_process.start()

    def Inference(self, req: grpc_def.ImageBatchRequest, context):
        if req.opt is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Request must contain inference options")
            return

        num_images = len(req.images)
        if num_images == 0:
            return
        # Get images from request
        image_names = []
        img: Image.Image
        image_tensors = []
        req_img: grpc_def.Image
        for req_img in req.images:
            try:
                img: Image.Image = Image.open(io.BytesIO(req_img.images_data))
                img = img.convert("RGB")
                image_shape = img.size
                for t in self.transforms:
                    img = t(img, None)
                    if isinstance(img, tuple):
                        img = img[0]
                image_tensors.append(img)
                image_names.append(req_img.name)
            except Exception as e:
                self.logger.warning(f"Failed to process image, skip: {e}")

        response_pipe, send_pipe = mp.Pipe(False)

        # send request to background process
        self.raw_image_queue.put((image_tensors, send_pipe))

        # wait for response from the background process
        response = response_pipe.recv()
        ret = grpc_def.InferenceResult()
        for (result, name) in zip(response, image_names):
            extracted_objects_in_one_image = InferenceService.extract_information_one(result, image_shape, name)
            result_in_one_image = grpc_def.ResultPerImage()
            for extracted in extracted_objects_in_one_image:

                detection = grpc_def.Detection()
                detection.confidence = extracted["score"]
                detection.category = extracted["label"]
                detection.cropped = bool(extracted["is_cropped"])
                rle = extracted["rle"]
                detection.rle.size.extend(rle["size"])
                detection.rle.counts = rle["counts"]

                bbox = extracted["bbox"]
                detection.bbox.xlt = bbox[0]
                detection.bbox.ylt = bbox[1]
                detection.bbox.xrb = bbox[2]
                detection.bbox.yrb = bbox[3]

                result_in_one_image.detections.append(detection)
                result_in_one_image.image_id = name
            ret.result.append(result_in_one_image)

        n = req.opt.num_image_returned
        if n == 0:
            pass
        elif n == -1:
            ret.returned_images.extend(req.images)
        else:
            ret.returned_images.extend(random.choices(req.images, k=n))
        return ret

    @staticmethod
    def inference_task(inference_queue: mp.Queue, batch_max_size: int):
        inference_service = InferenceService(cfg)

        while True:
            images: list
            images, send_pipe = inference_queue.get(True)
            send_pipes = [send_pipe]
            send_pipe_num_images = [len(images)]
            # get more
            while True:
                if len(images) >= batch_max_size:
                    break
                else:
                    try:
                        images_more, send_pipe_more = inference_queue.get(False)
                        images.extend(images_more)
                        send_pipes.append(send_pipe_more)
                        send_pipe_num_images.append(len(images_more))
                    except Exception:
                        break

            dataloader = DataLoader(
                MemoryFiles(images, None),
                shuffle=False,
                num_workers=1,
                batch_size=batch_max_size,
                collate_fn=MemoryFilesCollator(inference_service.cfg.DATALOADER.SIZE_DIVISIBILITY)
            )
            results = []
            for batch in dataloader:
                result = inference_service.run_inference(batch)
                results.extend(result)
            pipe: multiprocessing.connection.Connection
            for (pipe, num_imgs) in zip(send_pipes, send_pipe_num_images):
                pipe.send(results[:num_imgs])
                del results[:num_imgs]

    @staticmethod
    def serve():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ])
        grpc_service.add_InferenceServicer_to_server(InferenceServiceImpl(), server)
        server.add_insecure_port(f"{cfg.server_ip}:{cfg.server_port}")
        server.start()
        try:
            _ONE_DAY_IN_SECONDS = 60 * 60 * 24
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    InferenceServiceImpl.serve()

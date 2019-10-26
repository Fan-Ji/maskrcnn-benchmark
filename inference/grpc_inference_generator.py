import logging
import threading

import grpc
import io

import random
import torch
import torchvision
from PIL import Image
from inference.configuration import cfg

from inference.service import InferenceService

import inference.proto.inference_service_pb2_grpc as grpc_service
import inference.proto.inference_service_pb2 as grpc_def
import torch.multiprocessing as mp


# Consume client uploading stream, yield process results.
class GRPCInferenceGenerator(object):
    def __init__(self, request_iterator, context: grpc.ServicerContext):
        super(GRPCInferenceGenerator, self).__init__()
        self.context = context
        self.request_iterator = request_iterator
        self.meta = None
        self.logger = logging.getLogger("GRPCInferenceGenerator")
        self.raw_image_queue = mp.Queue()
        self.inference_result_queue = mp.Queue()


        self.request_grabber_thread = threading.Thread(target=self.request_iterator_grabber_task)

    def __iter__(self):
        return self

    def __next__(self):
        # return the processed image.
        # grab data from sync queue.
        # main thread blocks here.
        queue_result = self.inference_result_queue.get(True)
        for image_batch, result, image_name in queue_result:
            InferenceService.extract_information_one(result, self.meta["image_shape"], image_name)

        response = grpc_def.InferenceResult()

        return_images = []
        n = self.meta["num_image_returned"]
        if n == 0:
            pass
        elif n == -1:
            return_images = queue_result[0]
        else:
            return_images = random.choices(queue_result[0], k=n)

        for img, img_name in return_images:
            image_in_response = grpc_def.Image()
            image_in_response.name = img_name
            image_in_response.images_data = img
            response.returned_images.append(image_in_response)

    def request_iterator_grabber_task(self):
        to_tensor = torchvision.transforms.ToTensor()
        while True:
            req: grpc_def.ImageBatchRequest = next(self.request_iterator)
            if self.meta is None:
                if req.opt is None:
                    self.abort("First request must contain inference options")
                    return
                opt = req.opt
                self.meta["num_image_returned"] = opt.num_image_returned
                image_shape = (opt.image_width, opt.image_height)
                if opt.color_channel != 1:
                    image_shape += (opt.color_channel,)
                self.meta["image_shape"] = image_shape

                self.logger.info(f"Request initialized {self.meta}")
            num_images = len(req.images)
            if num_images == 0:
                continue

            # Get images from request
            images = []
            img: grpc_def.Image
            for img in req.images:
                try:
                    image_io = Image.open(io.BytesIO(img.images_data))
                    image_tensor = to_tensor(image_io)
                    images.append((image_tensor, img.name))
                except:
                    self.logger.warn("Failed to process image, skip")
            self.raw_image_queue.put(*images)


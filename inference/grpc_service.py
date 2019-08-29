import time

import grpc
from concurrent import futures

import sys
import logging
import os
import multiprocessing as mp
import io

from PIL import Image

import inference.proto.inference_service_pb2_grpc as grpc_service
import inference.proto.inference_service_pb2 as grpc_def
from inference.configuration import cfg
from inference.service import InferenceService

# logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InferenceServiceImpl")
logger.setLevel(logging.INFO)


class InferenceServiceImpl(grpc_service.InferenceServicer):
    def __init__(self):
        super(InferenceServiceImpl, self).__init__()
        self.inference_service = InferenceService(cfg, logger.getChild("Internal"))
        # self.inference_queue = mp.Queue()
        # self.inference_workers = [mp.Process(target=self.inference_worker_task)]
        # self.extraction_queue = mp.Queue()
        # self.extraction_workers = [mp.Process]
        # self.image_shape = (800,600) #TODO: hardcoded
        # self.info_extraction_queue = mp.Queue()
        # self.info_extraction_workers = mp.Pool(cfg.n_info_extraction_workers)
        # self.info_extraction_workers.apply()

    def StreamInference(self, request_iterator, context):
        meta = None
        req: grpc_def.ImageBatchRequest
        result_collection = []
        processed_images = 0
        for req in request_iterator:
            # First request: initialize options
            if meta is None:
                if req.opt is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("First request must contain inference options")
                    return

                meta = {}
                opt = req.opt
                meta["exclude_cropped"] = opt.exclude_cropped
                meta["area_dist"] = opt.area_dist
                meta["ellipse_dist"] = opt.ellipse_dist
                meta["image_batch_size"] = opt.image_batch_size
                logger.info(f"Request initialized {meta}")

            num_images = len(req.images)
            if num_images > 0:
                logger.info(f"Received {num_images}. Previously processed {processed_images} images")
            else:
                logger.info(f"No image received. Previously processed {processed_images} images")
                continue

            # Get images from request
            image_file_content = []
            for img in req.images:
                try:
                    image_file_content.append(io.BytesIO(img))
                except:
                    logger.warn("Failed to process image, skip")

            images = [Image.open(f) for f in image_file_content]
            # self.inference_queue.put(*images)
            results = self.inference_service.process(images)
            # TODO use workers to parallalize cpu processing
            # TODO decouple request to serve multiple clients!
            extracted_result = self.inference_service.extract_information(results, images[0].size, options=meta)
            result_collection.extend(extracted_result)
            processed_images += num_images
            if processed_images >= meta["image_batch_size"]:
                result = grpc_def.InferenceResult()
                result.result.processed_images = processed_images
                # TODO: result.sample_images
                for r in result_collection:
                    detection = grpc_def.Detection()
                    detection.area = r["area"]
                    result.result.detections.append(detection)
                logger.info("Sending back results")
                yield result
                processed_images = 0
                result_collection.clear()

    @staticmethod
    def serve():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        grpc_service.add_InferenceServicer_to_server(InferenceServiceImpl(), server)
        server.add_insecure_port(f"{cfg.server_ip}:{cfg.server_port}")
        server.start()
        try:
            _ONE_DAY_IN_SECONDS = 60 * 60 * 24
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


# def inference_worker_task(obj: InferenceServiceImpl):
#     while True:
#         imgs = []
#         trial_get = obj.inference_queue.get()
#         imgs.append(trial_get)
#         remaining = obj.inference_queue.qsize()
#         for i in range(remaining):
#             try:
#                 imgs.append(obj.inference_queue.get_nowait())
#             except:
#                 pass
#         results = obj.inference_service.process(imgs)
#         obj.extraction_queue.put(*results)
# def extraction_worker_task(obj: InferenceServiceImpl):
#     while True:
#         detections = obj.extraction_queue.get()
#         obj.inference_service.extract_information_one(detections, obj.image_shape,  )
if __name__ == '__main__':
    InferenceServiceImpl.serve()

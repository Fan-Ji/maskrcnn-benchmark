import time
import os
import random
import grpc
import inference.proto.inference_service_pb2_grpc as grpc_service
import inference.proto.inference_service_pb2 as grpc_def
from inference.configuration import cfg

N = 10
if __name__ == '__main__':
    channel = grpc.insecure_channel(f'localhost:{cfg.server_port}')
    stub = grpc_service.InferenceStub(channel)
    def gen_msg():
        while True:
            req = grpc_def.ImageBatchRequest()
            req.opt.num_image_returned = 1
            dir_path = "/home/wuyuanyi/nndata/images"
            imgs = os.listdir("/home/wuyuanyi/nndata/images")
            for ind in range(N):
                imgfiles = random.choice(imgs)
                image = grpc_def.Image()
                image.name = f"{ind}.png"
                with open(os.path.join(dir_path, imgfiles), 'rb') as f:
                    image.images_data = f.read()
                req.images.append(image)

            tic = time.time()
            resp = stub.Inference(req)
            toc = time.time()
            print((toc-tic)/N)


    gen_msg()
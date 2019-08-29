import time
import os
import random
import grpc
import inference.proto.inference_service_pb2_grpc as grpc_service
import inference.proto.inference_service_pb2 as grpc_def
from inference.configuration import cfg

if __name__ == '__main__':
    channel = grpc.insecure_channel(f'localhost:{cfg.server_port}')
    stub = grpc_service.InferenceStub(channel)
    def gen_msg():
        req = grpc_def.ImageBatchRequest()
        req.opt.image_batch_size = 50

        yield req
        while True:
            req = grpc_def.ImageBatchRequest()
            dir_path = "/home/wuyuanyi/nndata/images"
            imgs = os.listdir("/home/wuyuanyi/nndata/images")
            for ind in range(5):
                imgfiles = random.choice(imgs)
                with open(os.path.join(dir_path, imgfiles), 'rb') as f:
                    req.images.append(f.read())
            yield req
            time.sleep(1.2)

    resp = stub.StreamInference(gen_msg())
    for r in resp:
        print(r)
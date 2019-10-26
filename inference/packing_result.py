import os
import typing
import zipfile
import json
import datetime
import multiprocessing
import pickle
import os.path as osp

import io

from inference.configuration import cfg

# import inference.proto.inference_service_pb2 as grpc_def
from inference.proto.inference_service_pb2 import *

def _packing_result(cfg, data, filenames, inferences):
    filename = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.zip"
    with zipfile.ZipFile(osp.join(cfg.upload_store_dir, filename), 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        file: typing.TextIO
        for fn, d in zip(filenames, data):
            d.seek(0)
            zf.writestr(fn, d.read())

        if cfg.inference_store:
            zf.writestr('inference.pkl', pickle.dumps(inferences))


class ResultPacker:
    def __init__(self, cfg, threaded):
        self.cfg = cfg
        self.disabled = False
        if cfg.upload_store_dir is None:
            self.disabled = True

        os.makedirs(cfg.upload_store_dir, exist_ok=True)

        self.threaded = threaded

        if self.threaded:
            self.pool = multiprocessing.Pool()

    def packing_result(self, filenames, data, inferences):
        """
        save images to a zip archive
        :param filenames: list of strings (file names with ext of the images)
        :param data: list of image data stream to the encoded image buffer
        :param inferences:  grpc_def.InferenceResult object
        :param cfg: inference configuration
        :param threaded: use multiprocessing to do the work.
        :return:
        """

        if self.disabled:
            return

        if self.threaded:
            self.pool.apply_async(_packing_result, args=(self.cfg, data, filenames, inferences)).get()
        else:
            _packing_result(self.cfg, data, filenames, inferences)


if __name__ == '__main__':
    cfg.upload_store_dir = "/tmp/result_packer_test"
    cfg.inference_store = True

    packer = ResultPacker(cfg, True)

    filenames = ["test_data.png"]
    inference = InferenceResult(
        result=[
            ResultPerImage(image_id="test_data.png", detections=[Detection(
                rle=RLE(size=[800, 600], counts="//Test//Test====="),
                cropped=None,
                category=1,
                confidence=0.99,
                bbox=Rectangle(xlt=0, ylt=1, xrb=2, yrb=3)
            )])
        ]
    )
    data = [
        io.BytesIO(bytes("TestTestTestTest", "utf-8"))
    ]
    packer.packing_result(filenames, data, inference)
    packer.pool.close()
    packer.pool.join()

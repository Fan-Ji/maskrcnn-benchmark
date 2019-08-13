from inference.configuration import cfg
from flask import Flask, request, abort, jsonify, Response
from inference import service
from PIL import Image
from io import BytesIO
import logging
import os.path as osp

from inference.packing_result import packing_result


def main():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    inference_service = service.InferenceService(cfg, logger=app.logger)


    @app.route("/", methods=["POST"])
    def process():
        files = list(request.files.values())
        data = [BytesIO(f.read()) for f in files]
        original_filenames = [osp.basename(f.filename) for f in files]
        images = [Image.open(f) for f in data]

        message = {"result": inference_service.process(images, images[0].size, original_filenames)}

        packing_result(original_filenames, data, message, cfg, threaded=True)
        resp:Response = jsonify(message)

        return resp
    app.run(host=cfg.server_ip, port=cfg.server_port, debug=cfg.debug)


if __name__ == '__main__':
    main()
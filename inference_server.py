from inference.configuration import cfg
from flask import Flask, request, abort, jsonify, Response
from inference import service
from PIL import Image
from io import BytesIO
import logging
def main():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    inference_service = service.InferenceService(cfg, logger=app.logger)


    @app.route("/", methods=["POST"])
    def process():
        files = request.files.values()
        images = [Image.open(BytesIO(f.read())) for f in files]

        message = {"result": inference_service.process(images, images[0].size)}
        resp = jsonify(message)
        return resp
    app.run(host=cfg.server_ip, port=cfg.server_port)


if __name__ == '__main__':
    main()
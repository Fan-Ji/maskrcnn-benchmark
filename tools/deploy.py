import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="The output path.",
        default=None,
        required=True
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    model = build_detection_model(cfg)
    torch.save(model, args.output)
class Configuration(object):
    def __init__(self):
        # Model configurations
        self.config_file = "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"
        self.weight_file = "/home/wuyuanyi/nndata/train/model_final.pth"
        self.batch_size = 4
        self.n_cpu = 4
        self.nms_conf_thres = 0.5
        self.nms_thres = 0.5

        # Server configurations
        self.server_ip = "0.0.0.0"
        self.server_port = 3034

        # Postprocessing configurations
        self.cropped_threshold = 2

        # Statistics configuration
        self.window_size = 120

    def dict_override(self, override: dict):
        self.__dict__.update(override)

cfg = Configuration()
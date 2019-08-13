class Configuration(object):
    def __init__(self):
        # Model configurations
        self.config_file = "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"
        self.weight_file = "/home/wuyuanyi/nndata/train/model_final.pth"
        self.batch_size = 4
        self.n_cpu = 4

        # Server configurations
        self.server_ip = "0.0.0.0"
        self.server_port = 3034
        self.debug = False

        # Upload and inference storage. Will create a zip file for each upload. If inference_store is True, a inference json file will be included in the zip.
        self.upload_store_dir = "/home/wuyuanyi/nndata/maskrcnn_server_storage"  # None: do not store upload
        self.inference_store = True


        # Postprocessing configurations
        self.cropped_threshold = 2

    def dict_override(self, override: dict):
        self.__dict__.update(override)

cfg = Configuration()
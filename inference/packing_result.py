import typing
import zipfile
import json
import datetime
import threading

import os.path as osp

def packing_result(filenames, data, inferences, cfg, threaded):
    if cfg.upload_store_dir is None:
        return
    def _packing_result():
        filename = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.zip"
        with zipfile.ZipFile(osp.join(cfg.upload_store_dir, filename), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            file: typing.TextIO
            for fn, d in zip(filenames, data):
                d.seek(0)
                zf.writestr(fn, d.read())

            if cfg.inference_store:
                zf.writestr('result.json', json.dumps(inferences))

    if threaded:
        # Use thread instead of multiprocessing because the io operation does not lock the GIL.
        t = threading.Thread(target=_packing_result)
        t.start()
    else:
        _packing_result()

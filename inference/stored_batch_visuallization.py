import zipfile
from io import BytesIO
import argparse
import json
import cv2
import numpy as np
import math
import os
import os.path as osp
from pycocotools.coco import maskUtils as mask_utils
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("batch_zip", help="Zip file path")
    parser.add_argument("--output", help="Output directory. If not specified, dump files on the same directory as the zip file")
    parser.add_argument("--result-name", help="Inference result json name.", default="result.json")
    parser.add_argument("--no-box", help="Do not draw bounding box", action="store_true")
    parser.add_argument("--no-label", help="Do not draw label", action="store_true")
    parser.add_argument("--no-mask", help="Do not draw mask", action="store_true")
    parser.add_argument("--alpha", default=0.2)
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = osp.dirname(args.batch_zip)
        output = osp.join(output, "output")
        os.makedirs(output, exist_ok=True)
    with zipfile.ZipFile(args.batch_zip, "r") as zf:
        files = zf.namelist()

        name = args.result_name
        if not name in files:
            raise FileNotFoundError(f"The inference file {name} is not found in the archive")

        files.remove(name)
        dataset = json.loads(zf.read(name))
        results = dataset["result"]
        img_dict = dict()
        for result in results:
            img_entry = img_dict.get(result["img"])
            if img_entry is None:
                img_entry = []
                img_entry.append(result)
                img_dict[result["img"]] = img_entry
            else:
                img_entry.append(result)

        for f in files:
            # Check if annotation exists for this file.
            if not f in img_dict.keys():
                print(f"{f} is not found in the result file. Skip.")
                continue

            img_in_archive = zf.read(f)
            try:
                img_buf = np.asarray(bytearray(img_in_archive), dtype="uint8")
                im = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            except:
                print(f"{f} is not a valid image file. Skip.")
                continue

            anns = img_dict[f]
            pack = [(ann["mode"], ann["bbox"], ann["label"], ann["score"], ann["rle"]) for ann in anns]


            for m, b, l, s, r  in pack:
                b = [math.ceil(coor) for coor in b]
                if m == "xyxy":
                    pt1 = (b[0], b[1])
                    pt2 = (b[2], b[3])
                else:
                    pt1 = (b[0], b[1])
                    pt2 = (b[0] + b[2], b[1] + b[3])
                if not args.no_mask:
                    decoded_mask = mask_utils.decode(r)
                    decoded_mask_3ch = np.stack((decoded_mask*args.alpha,)*3, axis=-1)
                    green_dropback = np.zeros_like(im)
                    green_dropback[:,:,1] = 255

                    foreground = cv2.multiply(decoded_mask_3ch, green_dropback, dtype=cv2.CV_32FC3)
                    background = cv2.multiply(1-decoded_mask_3ch, im, dtype=cv2.CV_32FC3)

                    im = cv2.add(foreground, background)

                if not args.no_box:
                    cv2.rectangle(im, pt1, pt2, (255, 0, 0), thickness=2)
                if not args.no_label:
                    cv2.putText(im, f"{l}:{s}", pt1, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            print(f"Processed {f}")
            cv2.imwrite(osp.join(output, f), im)


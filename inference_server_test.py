import requests
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="server ip", default="127.0.0.1")
    parser.add_argument("--port", help="server port", default=3034)
    parser.add_argument("images", nargs='+')
    args = parser.parse_args()
    images = {}
    for img in args.images:
        with open(img, 'rb') as f:
            images[img] = f.read()
    print(f"Sending {len(images)} files")
    req = requests.post(f"http://{args.ip}:{args.port}", files = images)
    print(req.json())

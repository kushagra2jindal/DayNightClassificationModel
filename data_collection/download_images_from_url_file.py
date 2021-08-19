import requests
from requests import exceptions
from PIL import Image
import os
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="text file containing URLs")
ap.add_argument("-o", "--output", required=True,
    help="image download directory")
ap.add_argument("-ht", "--height", required=False,
    default=600, help=">height constraint")
ap.add_argument("-wd", "--width", required=False,
    default=600, help=">width constraint")
ap.add_argument("-s", "--status", required=False,
    default="True", help="show live download status")
ap.add_argument("-t", "--threads", required=False,
    default=1, help="number of downloader threads")
args = vars(ap.parse_args())


input_file = args["input"]
output_dir = args["output"]
max_threads = int(args["threads"])
height = int(args["height"])
width = int(args["width"])
show_status = True if args["status"] in ["yes", "true", "True", "1", "t", "y", "a", "+"] else False


assert max_threads>0, "ValueError: --threads {}".format(max_threads)
assert os.path.exists(input_file), "no such file {}".format(input_file)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

URLS = [line.rstrip("\n") for line in open(input_file)]
EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])


def download_images(*args):
    try:
        idx = int(args[0]["counter"] if type(args[0]) == dict else args[0])
        link = URLS[idx]

        r = requests.get(link, stream=True, allow_redirects=True, timeout=30)
        r.raise_for_status()
        r.raw.decode_content = True

        file_name = os.path.join(output_dir, str(idx)+".jpg")
        with Image.open(r.raw) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            w, h = img.size
            if h >= height or w >= width:
                img.save(file_name)

        if show_status: print("downloaded: {}".format(file_name))
        r.close()
    except Exception as e:
        if not type(e) in EXCEPTIONS:
            print(e)


print("downloading images")


if max_threads == 1:
    st = time.time()
    for i,link in enumerate(URLS):
        download_images({"counter":i})
    print("elapsed: {} sec".format(round((time.time()-st),3)))

else:
    import imthread
    st = time.time()
    imthread.start(download_images, repeat=len(URLS), max_threads=max_threads)
    print("elapsed: {} min".format(round(((time.time()-st)/60),3)))
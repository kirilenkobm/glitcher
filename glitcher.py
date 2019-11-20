#!/usr/bin/env python3
import random
import string
import os
import sys
import argparse
import subprocess
import numpy as np
from skimage import io
from skimage import transform as tf
from skimage import exposure

TEMP_DIR = "temp"
LAME_BINARY = "lame"
KHZ = 16
S_QUALITY = 3
BITRATE = 12
SHIFT = -580
GAMMA = 0.5
L_P = 10
R_P = 95


def parse_args():
    """Read and check args."""
    app = argparse.ArgumentParser()
    app.add_argument("input", help="Input image")
    app.add_argument("output", help="Output image")
    app.add_argument("--rgb_kt", "-r", default=16, type=int,
                     help="Aberrations power, 16 is default")
    if len(sys.argv) < 3:
        app.print_help()
        sys.exit(0)
    args = app.parse_args()
    if args.rgb_kt < 0:
        raise ValueError("Rgb shift parameter must be > 0!")
    return args


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    """Return random string for temp files."""
    return "".join(random.choice(chars) for _ in range(size))


def process_layer(layer):
    """Mp3 compress and decompress a layer."""
    w, h = layer.shape
    layer_flat = layer.reshape((w * h))
    flat_255 = np.around(layer_flat * 255, decimals=0)   
    flat_255[flat_255 > 255] = 255
    flat_255[flat_255 < 0] = 0
    int_form = list(map(int, flat_255))
    bytes_str = bytes(int_form)
    
    temp_files = []
    os.mkdir(TEMP_DIR) if not os.path.isdir(TEMP_DIR) else None
    raw_channel = os.path.join(TEMP_DIR, f"init_{id_gen()}.blob")
    mp3_compressed = os.path.join(TEMP_DIR, f"compr_{id_gen()}.mp3")
    decompressed = os.path.join(TEMP_DIR, f"decompr_{id_gen()}.blob")
    temp_files.extend([raw_channel, mp3_compressed, decompressed])

    with open(raw_channel, "wb") as f:
        f.write(bytes_str)

    mp3_compr = f'{LAME_BINARY} -r --unsigned -s {KHZ} -q {S_QUALITY} --resample 16 ' \
                f'--bitwidth 8 -b {BITRATE} -m m {raw_channel} "{mp3_compressed}"'
    mp3_decompr = f'{LAME_BINARY} --decode -x -t "{mp3_compressed}" {decompressed}'

    rc = subprocess.call(mp3_compr, shell=True)
    if rc != 0:
        sys.exit("Sorry, lame failed")

    rc = subprocess.call(mp3_decompr, shell=True)
    if rc != 0:
        sys.exit("Sorry, lame failed")

    with open(decompressed, "rb") as f:
        mp3_bytes = f.read()

    for t_file in temp_files:
        os.remove(t_file) if os.path.isfile(t_file) else None
    
    every_2nd = [x / 255 for i, x in enumerate(mp3_bytes[: 2 * len(bytes_str)]) if i % 2 == 0]
    result = np.array(every_2nd).reshape(w, h, 1)
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    return result


def adjust_contrast(im, l_p, r_p):
    """Contrast correction."""
    perc_left, perc_right = np.percentile(im, (l_p, r_p))
    im = exposure.rescale_intensity(im, in_range=(perc_left, perc_right))
    return im


def rgb_shift(img, kt):
    """Apply chromatic aberration."""
    if kt % 2 != 0:
        kt += 1
    shp = img.shape
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    # split channels, make shift
    red = tf.resize(red, output_shape=(shp[0], shp[1]))
    green = tf.resize(green, output_shape=(shp[0] - kt, shp[1] - kt))
    blue = tf.resize(blue, output_shape=(shp[0] - 2 * kt, shp[1] - 2 * kt))

    w, h = blue.shape
    ktd2 = int(kt / 2)
    red_n = np.reshape(red[kt: -kt, kt: -kt], (w, h, 1))
    green_n = np.reshape(green[ktd2: -1 * ktd2, ktd2: -1 * ktd2], (w, h, 1))
    blue_n = np.reshape(blue[:, :], (w, h, 1))

    new_im = np.concatenate((red_n, green_n, blue_n), axis=2)
    new_im = tf.resize(new_im, (shp[0], shp[1]))
    return new_im


def main():
    """Entry point."""
    args = parse_args()
    im = io.imread(args.input)
    im = tf.resize(im, (960, 1280))
    im = rgb_shift(im, kt=args.rgb_kt)
    im = exposure.adjust_gamma(image=im, gain=GAMMA)
    layers_upd = []
    for l_num in range(im.shape[2]):
        layer = im[:, :, l_num]
        proc_layer = process_layer(layer)
        layers_upd.append(proc_layer)
    upd_im = np.concatenate(layers_upd, axis=2)
    upd_im = np.roll(a=upd_im, axis=1, shift=SHIFT)
    upd_im = adjust_contrast(upd_im, L_P, R_P)
    io.imsave(args.output, upd_im)

if __name__ == "__main__":
    main()

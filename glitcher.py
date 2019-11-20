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
RGB_KT = 16


def parse_args():
    """Read and check args."""
    app = argparse.ArgumentParser()
    app.add_argument("input")
    app.add_argument("output")
    args = app.parse_args()
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


def horizonal_shifts(im, colorized=False):
    """Add random horizontal shifts."""
    w, h, d = im.shape
    processed = []
    shifts_borders_num = random.choice(range(6, 16, 2))
    shifts_borders = [0] + list(sorted(np.random.choice(range(w), shifts_borders_num, replace=False))) + [h]
    for num, border in enumerate(shifts_borders[1:]):
        prev_border = shifts_borders[num]
        pic_piece = im[prev_border: border, :, :]
        shift = 0 if num % 2 != 0 else random.choice(range(20))
        shifted = np.roll(pic_piece, shift=shift, axis=1)
        shifted = shifted if not colorized else np.roll(shifted, shift=shift, axis=2)
        processed.append(shifted)
    new_im = np.concatenate(processed, axis=0)
    new_im = tf.resize(new_im, (w, h))
    return new_im


def amplify(im):
    """Self-overlap."""
    REPEATS = 0
    shift = int(np.random.uniform(low=30, high=130))
    sign = np.random.choice([-1, 1], 1)[0]
    shift *= sign
    # print(shift)
    delim, kt = 1, 3
    layer_sh = np.roll(a=im, axis=1, shift=shift) / kt
    im += layer_sh
    delim += 1 / kt
    kt /= 3

    for _ in range(REPEATS):
        layer_sh = np.roll(a=layer_sh, axis=1, shift=shift) / kt
        im += layer_sh
        delim += 1 / kt
        kt /= 3

    im /= delim
    # im[im > 1] = 1.0
    return im


def main():
    """Entry point."""
    args = parse_args()
    im = io.imread(args.input)
    im = tf.resize(im, (960, 1280))
    im = rgb_shift(im, kt=RGB_KT)
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

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
from skimage import filters
from skimage.draw import polygon
from pydub import AudioSegment


TEMP_DIR = "temp"
LAME_BINARY = "lame"
KHZ = 16
SHIFT = 150


def parse_args():
    """Read and check args."""
    app = argparse.ArgumentParser()
    app.add_argument("input", help="Input image")
    app.add_argument("output", help="Output image")
    app.add_argument("--rgb_kt", "-r", default=16, type=int,
                     help="Aberrations power, 16 is default")
    app.add_argument("--lp", type=int, default=10,
                     help="Value in range [0..50]. The bigger, "
                           "the brighter lights are")
    app.add_argument("--rp", type=int, default=95,
                     help="Value in range [50..100]. The lesser, "
                          "the deeper shadows are")
    app.add_argument("--gamma", "-g", type=float, default=0.4,
                     help="Gamma correction")
    app.add_argument("--bitrate", "-b", type=int, default=12,
                     help="Compressed sound bitrate")
    app.add_argument("--s_quality", "-q", type=int, default=3,
                     help="Sound quality, val in range [0..10]")
    app.add_argument("--stripe", "--cs", action="store_true",
                     dest="stripe", help="Draw a crimson stripe")
    if len(sys.argv) < 3:
        app.print_help()
        sys.exit(0)
    args = app.parse_args()
    if args.rgb_kt < 0:
        raise argparse.ArgumentTypeError("Rgb shift parameter must be > 0!")
    if not 0 <= args.lp <= 50:
        raise argparse.ArgumentTypeError("l_p arg must be > 0 and < 50")
    if not 50 <= args.rp <= 100:
        raise argparse.ArgumentTypeError("l_p arg must be > 0 and < 50")
    if args.gamma < 0.0:
        raise argparse.ArgumentTypeError("gamma value must be >= 0.0!")
    if args.bitrate < 0:
        raise argparse.ArgumentTypeError("bitrate must be a positve value!")
    if not 0 <= args.s_quality <= 10:
        raise argparse.ArgumentTypeError("Sound quality must be in range [0..10]!")
    return args


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    """Return random string for temp files."""
    return "".join(random.choice(chars) for _ in range(size))


def parts(lst, n=25):
    """Split a list into a list of lists of len n."""
    return [lst[i:i + n] for i in iter(range(0, len(lst), n))]


def process_mp3(mp3_file, shape, chan_num=0):
    """Add glitches to a mp3 file itself."""
    # get size of image
    x, y = shape[0], shape[1]
    # read the mp3
    sound = AudioSegment.from_mp3(mp3_file)
    sound_array = sound.get_array_of_samples()[:x * y]

    for num, inds in enumerate(parts(list(range(len(sound_array))), n=x)):
        if num % 2 != 0:
            continue
        for inum, ind in enumerate(inds):
            if chan_num == 0:
                sound_array[ind] = sound_array[ind - x * num % 10]
            elif chan_num == 1:
                sound_array[ind] = sound_array[ind - x * num % 11]
            elif chan_num == 2:
                sound_array[ind] = sound_array[ind - x * num % 12]

    new_sound = sound._spawn(sound_array)
    new_sound.export(mp3_file, format='mp3')


def process_layer(layer, s_qual, bitrate, ch_num):
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

    mp3_compr = f'{LAME_BINARY} -r --unsigned -s {KHZ} -q {s_qual} --resample 16 ' \
                f'--bitwidth 8 -b {bitrate} -m m {raw_channel} "{mp3_compressed}"'
    mp3_decompr = f'{LAME_BINARY} --decode -x -t "{mp3_compressed}" {decompressed}'

    rc = subprocess.call(mp3_compr, shell=True)
    if rc != 0:
        sys.exit("Sorry, lame failed")

    process_mp3(mp3_compressed, layer.shape, ch_num)

    rc = subprocess.call(mp3_decompr, shell=True)
    if rc != 0:
        sys.exit("Sorry, lame failed")

    with open(decompressed, "rb") as f:
        mp3_bytes = f.read()

    for t_file in temp_files:
        os.remove(t_file) if os.path.isfile(t_file) else None
    
    arr_diff = len(mp3_bytes) // len(bytes_str)
    mp3_trimmed = enumerate(mp3_bytes[: arr_diff* len(bytes_str)])
    every_Nth = [x / 255 for i, x in mp3_trimmed if i % arr_diff == 0]
    result = np.array(every_Nth).reshape(w, h, 1)
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


def add_stripe(im):
    """Add a random stripe."""
    stripe = np.zeros((im.shape))
    width = np.random.choice(range(10, 200), 1)[0]
    poly = np.array(((0, 0),
                     (0, width),
                     (im.shape[0], width),
                     (im.shape[0], 0)))

    rr, cc = polygon(poly[:, 0], poly[:, 1], stripe.shape)
    stripe[rr, cc, 0] = 0.7
    stripe[rr, cc, 1] = 0.1
    stripe[rr, cc, 2] = 0.2
    stripe = np.roll(stripe, shift=np.random.choice(range(1000), 1)[0], axis=1)
    stripe = filters.gaussian(stripe, sigma=5, multichannel=True, mode='reflect', cval=0.6)
    im += stripe
    im[im > 1.0] = 1.0
    return im


def main():
    """Entry point."""
    args = parse_args()
    im = io.imread(args.input)
    im = tf.resize(im, (960, 1280))
    im = rgb_shift(im, kt=args.rgb_kt)
    im = add_stripe(im) if args.stripe else im
    im = exposure.adjust_gamma(image=im, gain=args.gamma)
    layers_upd = []
    for l_num in range(im.shape[2]):
        layer = im[:, :, l_num]
        proc_layer = process_layer(layer, args.s_quality, args.bitrate, l_num)
        layers_upd.append(proc_layer)
    upd_im = np.concatenate(layers_upd, axis=2)
    upd_im = np.roll(a=upd_im, axis=1, shift=SHIFT)
    upd_im = adjust_contrast(upd_im, args.lp, args.rp)
    io.imsave(args.output, upd_im)

if __name__ == "__main__":
    main()

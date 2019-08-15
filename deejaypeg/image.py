import numpy as np
from PIL import Image, ImageOps
import tempfile as tmp
import piexif
import piexif.helper
import json
import os


eps = np.finfo(np.float).eps

class Coder(object):
    def __init__(
        self,
        format='jpg',
        quality=100,
        qtable=None
    ):
        self.format = format
        self.quality = quality

        if qtable is not None:
            self.qtables = [qtable]
        else:
            self.qtables = None

    def encodedecode(self, X):
        """encode/decode on the fly"""
        image_file = tmp.NamedTemporaryFile(suffix='.' + self.format)
        self.encode(X, out=image_file.name)
        Y, _ = self.decode(image_file.name)
        file_size = os.path.getsize(image_file.name)

        image_file.close()
        return Y, file_size

    def encode(self, X, out, user_data={}, inverted=False):
        """
        Input is (nb_frames, nb_bins, nb_channels)
        Pillow takes (img_height, img_width, channels),
        so we reshape
            nb_frames -> img_width
            nb_bins --> img_height
            nb_channels -> channel
        """

        buf = np.squeeze(X)
        # swap frames and bins
        buf = buf.swapaxes(0, 1)
        # flip so that image is low_frequency at the bottom
        buf = buf[::-1, ...]

        if buf.ndim <= 2:
            img = Image.fromarray(buf.astype(np.int8), 'L')
        else:
            # stack channels into red and green, not using blue
            buf = np.concatenate((buf, np.zeros(buf.shape[:-1] + (1,))), -1)
            img = Image.fromarray(buf.astype(np.int8), 'RGB')

        user_data['inverted'] = inverted

        if inverted:
            # use white background
            img = ImageOps.invert(img)

        user_comment = piexif.helper.UserComment.dump(
            json.dumps(user_data)
        )
        exif_ifd = {
            piexif.ExifIFD.UserComment: user_comment,
        }
        exif_dict = {"Exif": exif_ifd}
        exif_bytes = piexif.dump(exif_dict)
        img.save(
            out,
            quality=self.quality,
            optimize=True,
            exif=exif_bytes,
            qtables=self.qtables,
            subsampling=0
        )

    def decode(self, path):
        return read(path)


def extract_qtable(path):
    with Image.open(path) as im:
        qtable = im.quantization

    return {int(k): list(v) for k, v in qtable.items()}


def read(path):
    img = Image.open(path)
    try:
        exif_data = img._getexif()
        user_json = piexif.helper.UserComment.load(
            exif_data[piexif.ExifIFD.UserComment]
        )
        user_data = json.loads(user_json)
    except (TypeError, AttributeError) as e:
        user_data = None

    if user_data['inverted']:
        # use white background
        img = ImageOps.invert(img)

    img = np.array(img).astype(np.uint8)
    # inverse flipped image
    img = img[::-1, ...]
    if img.ndim <= 2:
        return np.atleast_3d(img).swapaxes(0, 1), user_data
    else:
        img = img.swapaxes(0, 1)
        # select only red and blue channels
        return img[:, :, [0, 1]], user_data

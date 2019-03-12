import soundfile as sf
import argparse
import deejaypeg as djpg
import numpy as np


# values from wizard.txt in jpeg9-a src package.
standard_l_qtable = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
    )
    args = parser.parse_args()
    audio, rate = sf.read(args.input)
    mono = True
    if mono:
        audio = np.atleast_2d(np.mean(audio, axis=1)).T

    # set up modules
    tf = djpg.TF()
    ls = djpg.LogScaler()
    qt = djpg.Quantizer()
    im = djpg.Coder(format='jpg', qtable=standard_l_qtable)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)
    print("Xc", Xc.shape)
    # log scale
    Xs = ls.scale(Xc)
    print("Xs", Xs.shape)
    # quantize to 8bit
    Xq = qt.quantize(Xs)
    print("Xq", Xq.shape)
    # write as jpg image and save bounds values
    im.encode(
        Xq,
        "quantized_image.jpg",
        user_data={'bounds': ls.bounds.tolist()}
    )
    """
    inverse path
    """

    Xm_hat, user_data = im.decode("quantized_image.jpg")
    print(user_data['bounds'])
    print("decode", Xm_hat.shape)
    Xm_hat = qt.dequantize(Xm_hat)
    print("dequantize", Xm_hat.shape)
    Xm_hat = ls.unscale(Xm_hat, bounds=user_data['bounds'])
    print("unscale", Xm_hat.shape)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)
    sf.write("reconstruction.wav", audio_hat, rate)

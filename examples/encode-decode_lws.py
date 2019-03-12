import soundfile as sf
import argparse
import deejaypeg as djpg
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
    )
    args = parser.parse_args()
    audio, rate = sf.read(args.input)
    mono = False
    if mono:
        audio = np.atleast_2d(np.mean(audio, axis=1)).T

    # set up modules
    tf = djpg.TF(n_fft=4096, n_hopsize=1024)
    ls = djpg.LogScaler()
    qt = djpg.Quantizer()
    im = djpg.Coder(format='jpg', quality=100)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)

    W = (8, 8)
    mhop = (4, 4)
    import commonfate
    # do transform it to a tensor but do NOT make the 2d-FFT
    x = commonfate.transform.split(Xc, W, mhop, True)
    

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
        user_data={
            'bounds': ls.bounds.tolist(),
            'n_fft': tf.n_fft,
            'n_hopsize': tf.n_hopsize
        }
    )
    """
    inverse path
    """

    Xm_hat, user_data = im.decode("quantized_image.jpg")
    print(user_data['bounds'])
    print("decode", Xm_hat.shape)
    Xm_hat = qt.dequantize(Xm_hat)
    print("dequantize", Xm_hat.shape)
    # Xm_hat = ls.unscale(Xm_hat)
    Xm_hat = np.abs(Xc)
    print("unscale", Xm_hat.shape)

    # use reconstruction with original phase
    lws_processor = lws.lws(4096, 1024, mode="music")
    X1l = lws_processor.run_lws(Xm_hat[..., 0])
    X1r = lws_processor.run_lws(Xm_hat[..., 1])
    X1 = np.transpose(np.stack([X1l, X1r]), (1, 2, 0))
    audio_hat = tf.inverse_transform(X1)
    sf.write("reconstruction.wav", audio_hat, rate)

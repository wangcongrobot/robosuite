# Orignal author: Roma Sokolkov
# VAE controller for runtime optimization.


import numpy as np

from robosuite.models.vae.model import ConvVAE
from robosuite.models.vae.data_loader import denormalize, preprocess_input
from robosuite.models.vae.config import IMAGE_WIDTH, IMAGE_HEIGHT, ROI

class VAEController:
    """
    Wrapper to manipulate a VAE.

    :param z_size: (int) latent space dimension
    :param input_dimension: ((int, int, int)) input dimension
    :param learning_rate: (float)
    :param kl_tolerance: (float) Clip the KL loss
        max_kl_loss = kl_tolerance * z_size
    :param batch_size: (int)
    :param normalization_mode: (str)
    """
    def __init__(self, z_size=None, 
                input_dimension=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                 learning_rate=0.0001, kl_tolerance=0.5,
                 batch_size=64, normalization_mode='rl'):
        # VAE input and output shapes
        self.z_size = z_size
        self.input_dimension = input_dimension

        # VAE params
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance

        # Training params
        self.batch_size = batch_size
        self.normalization_mode = normalization_mode

        self.vae = None
        self.target_vae = None

        if z_size is not None:
            self.vae = ConvVAE(z_size=self.z_size,
                               batch_size=self.batch_size,
                               learning_rate=self.learning_rate,
                               kl_tolerance=self.kl_tolerance,
                               is_training=True,
                               reuse=False)

            self.target_vae = ConvVAE(z_size=self.z_size,
                                      batch_size=1,
                                      is_training=False,
                                      reuse=False)

    def encode_from_raw_image(self, raw_image):
        """
        :param raw_image: (np.ndarray) BGR image
        """
        r = ROI
        print("\n\nROI: \n\n", r)
        # Crop image
        im = raw_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        print("im: \n\n", im.shape)
        return self.encode(im)

    def encode(self, observation):
        assert observation.shape == self.input_dimension, "{} != {}".format(observation.shape, self.input_dimension)
        # Normalize
        observation = preprocess_input(observation.astype(np.float32),
                                       mode=self.normalization_mode)[None]
        return self.target_vae.encode(observation)

    def decode(self, arr):
        assert arr.shape == (1, self.z_size), "{} != {}".format(arr.shape, (1, self.z_size))
        # Decode
        arr = self.target_vae.decode(arr)
        # Denormalize
        arr = denormalize(arr, mode=self.normalization_mode)
        return arr

    def save(self, path):
        self.target_vae.save(path)

    def load(self, path):
        self.target_vae = ConvVAE.load(path)
        self.z_size = self.target_vae.z_size

    def set_target_params(self):
        params = self.vae.get_params()
        self.target_vae.set_params(params)

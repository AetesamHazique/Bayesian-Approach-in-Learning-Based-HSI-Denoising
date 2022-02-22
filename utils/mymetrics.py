import tensorflow as tf
import keras.backend as K

def ssim_metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def psnr_metric(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

# Loss functtion
def ssim_loss(y_true, y_pred):
    return tf.reduce_mean((1.0-tf.image.ssim(y_true, y_pred,1))/2.0)

#def SSIMLoss(y_true, y_pred):
#  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


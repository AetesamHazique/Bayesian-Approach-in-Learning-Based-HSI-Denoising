import numpy as np
import scipy.signal
import caffe2

class SSIM(caffe.Layer):
    "A loss layer that calculates (1-SSIM) loss. Assuming bottom[0] is output data and bottom[1] is label, meaning no back-propagation to bottom[1]."

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.C1 = params.get('C1', 0.01) ** 2
        self.C2 = params.get('C2', 0.03) ** 2
        self.sigma = params.get('sigma', 5.)

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        if (bottom[0].width%2) != 1 or (bottom[1].width%2) != 1 :
            raise Exception("Odd patch size preferred")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

		# initialize the gaussian filter based on the bottom size
        width = bottom[0].width
        self.w = np.exp(-1.*np.arange(-(width/2), width/2+1)**2/(2*self.sigma**2))
        self.w = np.outer(self.w, self.w.reshape((width, 1)))	# extend to 2D
        self.w = self.w/np.sum(self.w)							# normailization
        self.w = np.reshape(self.w, (1, 1, width, width)) 		# reshape to 4D
        self.w = np.tile(self.w, (bottom[0].num, 3, 1, 1))

    def forward(self, bottom, top):
        self.mux = np.sum(self.w * bottom[0].data, axis=(2,3), keepdims=True)
        self.muy = np.sum(self.w * bottom[1].data, axis=(2,3), keepdims=True)
        self.sigmax2 = np.sum(self.w * bottom[0].data ** 2, axis=(2,3), keepdims=True) - self.mux **2
        self.sigmay2 = np.sum(self.w * bottom[1].data ** 2, axis=(2,3), keepdims=True) - self.muy **2
        self.sigmaxy = np.sum(self.w * bottom[0].data * bottom[1].data, axis=(2,3), keepdims=True) - self.mux * self.muy
        self.l = (2 * self.mux * self.muy + self.C1)/(self.mux ** 2 + self.muy **2 + self.C1)
        self.cs = (2 * self.sigmaxy + self.C2)/(self.sigmax2 + self.sigmay2 + self.C2)

        top[0].data[...] = 1 - np.sum(self.l * self.cs)/(bottom[0].channels * bottom[0].num)

    def backward(self, top, propagate_down, bottom):
        self.dl = 2 * self.w * (self.muy - self.mux * self.l) / (self.mux**2 + self.muy**2 + self.C1)
        self.dcs = 2 / (self.sigmax2 + self.sigmay2 + self.C2) * self.w * ((bottom[1].data - self.muy) - self.cs * (bottom[0].data - self.mux))

        bottom[0].diff[...] = -(self.dl * self.cs + self.l * self.dcs)/(bottom[0].channels * bottom[0].num)	    # negative sign due to -dSSIM
        bottom[1].diff[...] = 0


def contro_loss(self):
        '''
        总结下来对比损失的特点：首先看标签，然后标签为1是正对，负对部分损失为0，最小化总损失就是最小化类内损失(within_loss)部分，
        让s逼近margin的过程，是个增大的过程；标签为0是负对，正对部分损失为0，最小化总损失就是最小化between_loss，而且此时between_loss就是s，
        所以这个过程也是最小化s的过程，也就使不相似的对更不相似了
        '''
        s = self.similarity
        one = tf.constant(1.0)
        margin = 1.0
        y_true = tf.to_float(self.y_true)
 
        # 类内损失：
        max_part = tf.square(tf.maximum(margin-s,0)) # margin是一个正对该有的相似度临界值
        within_loss = tf.multiply(y_true,max_part) #如果相似度s未达到临界值margin，则最小化这个类内损失使s逼近这个margin，增大s
 
        # 类间损失：
        between_loss = tf.multiply(one-y_true,s) #如果是负对，between_loss就等于s，这时候within_loss=0，最小化损失就是降低相似度s使之更不相似
        
        # 总体损失（要最小化）：
        loss = 0.5*tf.reduce_mean(within_loss+between_loss)
        return loss

class Model:
  def __init__(self,batch_size):
    self.batch_size = batch_size
    
    def loss_DSSIS_tf11(self, y_true, y_pred):
        """Need tf0.11rc to work"""
        y_true = tf.reshape(y_true, [self.batch_size] + get_shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [self.batch_size] + get_shape(y_pred)[1:])
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
        patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

        u_true = K.mean(patches_true, axis=3)
        u_pred = K.mean(patches_pred, axis=3)
        var_true = K.var(patches_true, axis=3)
        var_pred = K.var(patches_pred, axis=3)
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim /= denom
        ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
        return K.mean(((1.0 - ssim) / 2))



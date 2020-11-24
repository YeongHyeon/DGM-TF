import numpy as np
import tensorflow as tf

class Layers(object):

    def __init__(self, parameters={}):

        self.num_params = 0
        self.initializer, self.parameters = {}, parameters

    def __initializer_random(self, shape, name=''):

        try: return self.initializer[name]
        except:
            try: initializer = tf.compat.v1.constant_initializer(self.parameters[name])
            except:
                try: stddev = np.sqrt(2/(shape[-2]+shape[-1]))
                except: stddev = np.sqrt(2/shape[-1])
                self.initializer[name] = tf.compat.v1.random_normal_initializer(\
                    mean=0.0, stddev=stddev, dtype=tf.dtypes.float32)
            return self.initializer[name]

    def __initializer_constant(self, shape, constant=0, name=''):

        try: return self.initializer[name]
        except:
            try: self.initializer[name] = tf.compat.v1.constant_initializer(self.parameters[name])
            except: self.initializer[name] = tf.compat.v1.constant_initializer(np.ones(shape)*constant)
            return self.initializer[name]

    def __get_variable(self, shape, constant=None, trainable=True, name=''):

        try: return self.parameters[name]
        except:
            try: initializer = self.__initializer_constant(shape=shape, constant=constant, name=name)
            except: initializer = self.__initializer_random(shape=shape, name=name)

            tmp_num = 1
            for num in shape:
                tmp_num *= num
            self.num_params += tmp_num
            self.parameters[name] = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=initializer, trainable=trainable)

            return self.parameters[name]

    def activation(self, x, activation=None, name=''):

        if(activation is None): return x
        elif("sigmoid" == activation):
            return tf.compat.v1.nn.sigmoid(x, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            return tf.compat.v1.nn.tanh(x, name='%s_tanh' %(name))
        elif("relu" == activation):
            return tf.compat.v1.nn.relu(x, name='%s_relu' %(name))
        elif("lrelu" == activation):
            return tf.compat.v1.nn.leaky_relu(x, name='%s_lrelu' %(name))
        elif("elu" == activation):
            return tf.compat.v1.nn.elu(x, name='%s_elu' %(name))
        else: return x

    def maxpool(self, x, ksize, strides, padding, name='', verbose=True):

        y = tf.compat.v1.nn.max_pool(value=x, \
            ksize=ksize, strides=strides, padding=padding, name=name)

        if(verbose): print("Pool", x.shape, "->", y.shape)
        return y

    def batch_normalization(self, x, trainable=True, training=None, name='', verbose=True):

        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/layers/batch_normalizationalization
        shape_in = x.get_shape().as_list()[-1]

        beta = self.__initializer_constant(shape=[shape_in], constant=0.0, name='%s_beta' %(name))
        gamma = self.__initializer_constant(shape=[shape_in], constant=1.0, name='%s_gamma' %(name))
        mv_mean = self.__initializer_constant(shape=[shape_in], constant=0.0, name='%s_mv_mean' %(name))
        mv_var = self.__initializer_constant(shape=[shape_in], constant=1.0, name='%s_mv_var' %(name))

        y = tf.compat.v1.layers.batch_normalization(inputs=x, \
            beta_initializer=beta,
            gamma_initializer=gamma,
            moving_mean_initializer=mv_mean,
            moving_variance_initializer=mv_var, \
            training=training, trainable=trainable, name=name)

        if(verbose): print("BN", x.shape, ">", y.shape)
        return y

    def conv2d(self, x, stride, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, training=None, activation=None, name='', verbose=True):

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-1]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.compat.v1.nn.conv2d(
            input=x,
            filter=w,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("Conv", x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, training=training, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def convt2d(self, x, stride, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, training=None, activation=None, name='', verbose=True):

        for idx_os, _ in enumerate(output_shape):
            if(idx_os == 0): continue
            output_shape[idx_os] = int(output_shape[idx_os])

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-2]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.compat.v1.nn.conv2d_transpose(
            value=x,
            filter=w,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("ConvT", x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, training=training, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def fully_connected(self, x, c_out, \
        batch_norm=False, training=None, activation=None, name='', verbose=True):

        c_in, c_out = x.get_shape().as_list()[-1], int(c_out)

        w = self.__get_variable(shape=[c_in, c_out], \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[c_out], \
            trainable=True, name='%s_b' %(name))

        wx = tf.compat.v1.matmul(x, w, name='%s_mul' %(name))
        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("FC", x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, training=training, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

import os
import numpy as np
import tensorflow as tf
import source.layers as lay

class DGM(object):

    def __init__(self, \
        height, width, channel, ksize, zdim, \
        learning_rate=1e-3, path='', verbose=True):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel, self.ksize, self.zdim = \
            height, width, channel, ksize, zdim
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel], \
            name="x")
        self.y = tf.compat.v1.placeholder(tf.float32, [None, 1], \
            name="x")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="batch_size")
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[], \
            name="training")

        self.layer = lay.Layers()

        self.variables, self.losses = {}, {}
        self.__build_model(x=self.x, ksize=self.ksize, verbose=verbose)
        self.__build_loss()

        with tf.control_dependencies(self.variables['ops_d']):
            self.optimizer_d = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, name='Adam_d').minimize(\
                self.losses['loss_d'], var_list=self.variables['params_d'])

        with tf.control_dependencies(self.variables['ops_g']):
            self.optimizer_g = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate*5, name='Adam_g').minimize(\
                self.losses['loss_g'], var_list=self.variables['params_g'])

        tf.compat.v1.summary.scalar('DGM/loss_a', self.losses['loss_a'])
        tf.compat.v1.summary.scalar('DGM/loss_r', self.losses['loss_r'])
        tf.compat.v1.summary.scalar('DGM/loss_tv', self.losses['loss_tv'])
        tf.compat.v1.summary.scalar('DGM/loss_g', self.losses['loss_g'])
        tf.compat.v1.summary.scalar('DGM/loss_d', self.losses['loss_d'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.__init_session(path=self.path_ckpt)

    def step(self, x, y, iteration=0, training=False):

        feed_tr = {self.x:x, self.y:y, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.y:y, self.batch_size:x.shape[0], self.training:False}

        summary_list = []
        if(training):
            try:
                _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)

                _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)
            except:
                _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)

                _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)

            for summaries in summary_list:
                self.summary_writer.add_summary(summaries, iteration)

        y_hat, loss_d, loss_g, mse = \
            self.sess.run([self.variables['y_hat'], self.losses['loss_d'], self.losses['loss_g'], self.losses['mse']], \
            feed_dict=feed_te)

        outputs = {'y_hat':y_hat, 'loss_d':loss_d, 'loss_g':loss_g, 'mse':mse}
        return outputs

    def save_parameter(self, model='model_checker', epoch=-1):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        if(epoch >= 0): self.summary_writer.add_run_metadata(self.run_metadata, 'epoch-%d' % epoch)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, verbose=True):

        print("\n* Parameter arrange")

        ftxt = open("list_parameters.txt", "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def confirm_bn(self, verbose=True):

        print("\n* Confirm Batch Normalization")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            if('bn' in var.name):
                tmp_x = np.zeros((1, self.height, self.width, self.channel))
                tmp_y = np.zeros((1, 1))
                values = self.sess.run(var, \
                    feed_dict={self.x:tmp_x, self.y:tmp_y, self.batch_size:1, self.training:False})
                if(verbose): print(var.name, var.shape)
                if(verbose): print(values)

    def loss_l1(self, x, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.abs(x), axis=reduce)

        return distance

    def loss_l2(self, x, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-9), axis=reduce)

        return distance

    def __init_session(self, path):

        try:
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
        except: pass

    def __build_loss(self):

        # Adversarial loss
        least_square_term = tf.math.square(\
            self.variables['d_real'] - tf.ones_like(self.variables['d_real']) + 1e-9)
        self.losses['loss_a'] = tf.compat.v1.reduce_mean(0.5 * least_square_term)

        # Reconstruction losses
        self.losses['loss_r1'] = \
            self.loss_l1(self.variables['y_hat'] + self.variables['a'] - self.x, [1, 2, 3])
        self.losses['loss_r2'] = \
            self.loss_l1(self.variables['z_phat'] - self.x, [1, 2, 3])

        y_ext = tf.compat.v1.reshape(self.y, shape=[self.batch_size, 1, 1, 1], \
                name="y_ext")
        self.losses['loss_r3'] = \
            self.loss_l1(self.variables['a'] * y_ext, [1, 2, 3])

        # Total variation loss
        shift_i = tf.roll(self.variables['y_hat'], shift=[1], axis=[1])
        shift_j = tf.roll(self.variables['y_hat'], shift=[1], axis=[2])
        tv_i_term = \
            self.loss_l1(self.variables['y_hat'] - shift_i, [1, 2, 3])
        tv_j_term = \
            self.loss_l1(self.variables['y_hat'] - shift_j, [1, 2, 3])
        self.losses['loss_tv'] = tv_i_term + tv_j_term

        lambda_a, lambda_r, lambda_tv = 10, 10, 1 # in paper
        term_a = lambda_a * self.losses['loss_a']
        term_r = lambda_r * \
            (self.losses['loss_r1'] + self.losses['loss_r2'] + self.losses['loss_r3'])
        term_tv = lambda_tv * self.losses['loss_tv']
        self.losses['loss_g'] = \
            tf.compat.v1.reduce_mean(term_a + term_r + term_tv)
        self.losses['loss_a'] = tf.compat.v1.reduce_mean(term_a)
        self.losses['loss_r'] = tf.compat.v1.reduce_mean(term_r)
        self.losses['loss_tv'] = tf.compat.v1.reduce_mean(term_tv)

        # For D optimization
        d_real_term = tf.math.square(self.variables['d_real'] + 1e-9)
        d_fake_term = tf.math.square(\
            self.variables['d_fake'] - tf.ones_like(self.variables['d_fake']) + 1e-9)
        self.losses['loss_d'] = tf.compat.v1.reduce_mean(0.5 * (d_real_term + d_fake_term))

        self.losses['mse'] = \
            tf.compat.v1.reduce_mean(self.loss_l2(self.variables['y_hat'] - self.x, [1, 2, 3]))

        self.variables['params_d'], self.variables['params_g'] = [], []
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if('dis_' in var.name): self.variables['params_d'].append(var)
            else: self.variables['params_g'].append(var)

        self.variables['ops_d'], self.variables['ops_g'] = [], []
        for ops in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS):
            if('dis_' in ops.name): self.variables['ops_d'].append(ops)
            else: self.variables['ops_g'].append(ops)

    def __build_model(self, x, ksize=3, verbose=True):

        print("\n-*-*- Flow 1 -*-*-")
        # Module 1: E_G
        self.variables['c_z'] = \
            self.__encoder(x=x, ksize=ksize, outdim=self.zdim, reuse=False, \
            name='enc_g', norm=True, verbose=verbose)
        # Module 2: D_G
        self.variables['y_hat'] = \
            self.__decoder(z=self.variables['c_z'], ksize=ksize, reuse=False, \
            name='gen_g', norm=True, verbose=verbose)
        # Module 3: D
        self.variables['d_real'] = \
            self.__encoder(x=x, ksize=ksize, outdim=1, reuse=False, \
            name='dis_g', norm=True, verbose=verbose)
        # Revisit D
        self.variables['d_fake'] = \
            self.__encoder(x=self.variables['y_hat'], ksize=ksize, outdim=1, reuse=True, \
            name='dis_g', norm=True, verbose=False)

        print("\n-*-*- Flow 2 -*-*-")
        # Module 4: E_F
        self.variables['c_s'] = \
            self.__encoder(x=x, ksize=ksize, outdim=self.zdim, reuse=False, \
            name='enc_f', norm=False, verbose=verbose)
        # Module 5: D_F
        self.variables['a'] = \
            self.__decoder(z=self.variables['c_s'], ksize=ksize, reuse=False, \
            name='gen_f', norm=True, verbose=verbose)

        print("\n-*-*- Flow 3 -*-*-")
        # Vector joint
        self.variables['c_joint'] = \
            tf.concat([self.variables['c_z'], self.variables['c_s']], 1)
        # Module 6: D_J
        self.variables['z_phat'] = \
            self.__decoder(z=self.variables['c_joint'], ksize=ksize, reuse=False, \
            name='gen_j', norm=True, verbose=verbose)


    def __encoder(self, x, ksize=3, outdim=1, reuse=False, \
        name='enc', activation='relu', norm=True, depth=3, verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            c_in, c_out = self.channel, 16
            for idx_d in range(depth):
                conv1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_in, c_out], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose)
                conv2 = self.layer.conv2d(x=conv1, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_out, c_out], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_2" %(name, idx_d), verbose=verbose)
                maxp = self.layer.maxpool(x=conv2, ksize=2, strides=2, padding='SAME', \
                    name="%s_pool%d" %(name, idx_d), verbose=verbose)

                if(idx_d < (depth-1)): x = maxp
                else: x = conv2

                c_in = c_out
                c_out *= 2

            rs = tf.compat.v1.reshape(x, shape=[self.batch_size, int(7*7*64)], \
                name="%s_rs" %(name))
            e = self.layer.fully_connected(x=rs, c_out=outdim, \
                batch_norm=False, training=self.training, \
                activation=None, name="%s_fc1" %(name), verbose=verbose)

            return e

    def __decoder(self, z, ksize=3, reuse=False, \
        name='dec', activation='relu', norm=True, depth=3, verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            c_in, c_out = 64, 64
            h_out, w_out = 14, 14

            fc1 = self.layer.fully_connected(x=z, c_out=7*7*64, \
                batch_norm=norm, training=self.training, \
                activation=activation, name="%s_fc1" %(name), verbose=verbose)
            rs = tf.compat.v1.reshape(fc1, shape=[self.batch_size, 7, 7, 64], \
                name="%s_rs" %(name))

            x = rs
            for idx_d in range(depth):
                if(idx_d == 0):
                    convt1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                        filter_size=[ksize, ksize, c_in, c_out], batch_norm=norm, training=self.training, \
                        activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose)
                else:
                    convt1 = self.layer.convt2d(x=x, stride=2, padding='SAME', \
                        output_shape=[self.batch_size, h_out, w_out, c_out], filter_size=[ksize, ksize, c_out, c_in], \
                        dilations=[1, 1, 1, 1], batch_norm=norm, training=self.training, \
                        activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose)
                    h_out *= 2
                    w_out *= 2

                convt2 = self.layer.conv2d(x=convt1, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_out, c_out], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_2" %(name, idx_d), verbose=verbose)
                x = convt2

                if(idx_d == 0):
                    c_out /= 2
                else:
                    c_in /= 2
                    c_out /= 2

            d = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, c_in, self.channel], batch_norm=False, training=self.training, \
                activation='sigmoid', name="%s_conv%d_3" %(name, idx_d), verbose=verbose)

            return d

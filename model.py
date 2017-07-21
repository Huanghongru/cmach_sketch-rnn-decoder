# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sketch-RNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# internal imports

import numpy as np
import tensorflow as tf

import rnn


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = tf.contrib.training.HParams(
        data_set=['cow.npz', 'crab.npz'],  # Our dataset.
        # num_steps=10000000,  # Total number of steps of training. Keep large.
        num_steps=2,
        save_every=1,  # Number of batches per checkpoint creation.
        max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
        dec_rnn_size=128,  # Size of decoder.
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper.
        enc_rnn_size=64,  # Size of encoder.
        enc_model='layer_norm',  # Encoder: lstm, layer_norm or hyper.
        z_size=32,  # Size of latent vector z. Recommend 32, 64 or 128.
        kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
        kl_weight_start=0.01,  # KL start weight when annealing.
        kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL. default 0.2
        batch_size=100,  # Minibatch size. Recommend leaving at 100.
        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
        num_mixture=20,  # Number of mixtures in Gaussian mixture model.
        learning_rate=0.001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        kl_decay_rate=0.9995,  # KL annealing decay rate per minibatch.
        min_learning_rate=0.0001,  # Minimum learning rate.
        use_recurrent_dropout=False,  # Dropout with memory loss. Recomended
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.90,  # Probability of input dropout keep.
        use_output_dropout=False,  # Output droput. Recommend leaving False.
        output_dropout_prob=0.90,  # Probability of output dropout keep.
        random_scale_factor=0.15,  # Random scaling data augmention proportion.
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
        conditional=True,  # When False, use unconditional decoder-only model.
        is_training=True,  # Is model training? Recommend keeping true.
        num_enc_experts=1   # The num of the encoder Experts
    )
    return hparams


class Model(object):
    """Define a SketchRNN model."""

    def __init__(self, hps, gpu_mode=True, reuse=False):
        """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
        self.hps = hps
        with tf.variable_scope('vector_rnn', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self.build_model()
            else:
                tf.logging.info('Model using gpu.')
                self.build_model()

    def encoder(self, batch, sequence_lengths, N):
        """Define the bi-directional encoder module of sketch-rnn.

    Args:
        batch: a 3d tensor. a batch of inputs, though the codes write "output_x"
        sequence_lengths: an int specified by training process, indicating sequence lengths of all input samples
        N: number of experts
    Return:
        mu1, presig1, mu2, presig2: parameters for the definition of latent vector z
    """
        unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            batch,
            sequence_length=sequence_lengths,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32,
            scope='ENC_RNN')

        last_state_fw, last_state_bw = last_states
        last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
        last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
        last_h = tf.concat([last_h_fw, last_h_bw], 1)

        # two bidirectional RNNs, so they do not share parameters
        params = []

        # TODO: are the params shared by iterations?
        with tf.variable_scope('ENC_RNN'):
            for i in range(N):
                scope_mu = "%s%d" % ("mu", i)
                scope_presig = "%s%d" % ("presig", i)

                mu = rnn.super_linear(
                    last_h,
                    self.hps.z_size,
                    input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
                    scope=scope_mu,
                    init_w='gaussian',
                    weight_start=0.001)

                presig = rnn.super_linear(
                    last_h,
                    self.hps.z_size,
                    input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
                    scope=scope_presig,
                    init_w='gaussian',
                    weight_start=0.001)

                params.append([mu, presig])

        return params

    def decoder(self, actual_input_x, initial_state):
        self.num_mixture = self.hps.num_mixture

        # Number of outputs is end_of_stroke + prob + 2*(mu + sig) + corr
        n_out = (3 + self.num_mixture * 6)

        with tf.variable_scope('DEC_RNN') as scope:
            try:
                output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
                output_b = tf.get_variable('output_b', [n_out])
            except ValueError:
                scope.reuse_variables()
                output_w = tf.get_variable('output_w')
                output_b = tf.get_variable('output_b')

            # decoder module of sketch-rnn is below
            output, last_state = tf.nn.dynamic_rnn(
                self.cell,
                actual_input_x,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32)

            # reshape output
            output = tf.reshape(output, [-1, self.hps.dec_rnn_size])
            output = tf.nn.xw_plus_b(output, output_w, output_b)

        # params of the decoder
        # should be saved
        out = self.get_mixture_coef(output)
        return out, last_state

    def tf_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
        """Returns result of eq # 24 and 25 of http://arxiv.org/abs/1308.0850."""
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
             2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result

    def get_lossfunc(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data):
        """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
        result0 = self.tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        epsilon = 1e-6
        result1 = tf.multiply(result0, z_pi)
        result1 = tf.reduce_sum(result1, 1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # avoid log(0)

        fs = 1.0 - pen_data[:, 2]  # use training data for this
        fs = tf.reshape(fs, [-1, 1])
        result1 = tf.multiply(result1, fs)

        result2 = tf.nn.softmax_cross_entropy_with_logits(
            labels=pen_data, logits=z_pen_logits)
        result2 = tf.reshape(result2, [-1, 1])
        if not self.hps.is_training:  # eval mode, mask eos columns
            result2 = tf.multiply(result2, fs)  # ? why remove eos result in eval mode ?

        result = result1 + result2
        return result

    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(self, output):
        """Returns the tf slices containing mdn dist params."""
        # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
        z = output
        z_pen_logits = z[:, 0:3]  # pen states, TODO: the pre-prob of pen states? not quite useful
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

        # process output z's into MDN paramters

        # softmax all the pi's and pen states:
        z_pi = tf.nn.softmax(z_pi)
        z_pen = tf.nn.softmax(z_pen_logits)  # TODO: z_pen? the prob of pen states

        # exponentiate the sigmas and also make corr between -1 and 1.
        z_sigma1 = tf.exp(z_sigma1)
        z_sigma2 = tf.exp(z_sigma2)
        z_corr = tf.tanh(z_corr)

        r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
        return r

    # get the inputs for decoder
    def compute_inputs(self, mu, presig):
        sigma = tf.exp(presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
        eps = tf.random_normal(
            (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        batch_z = mu + tf.multiply(sigma, eps)

        # KL cost
        kl_costs = -0.5 * tf.reduce_mean((1 + presig - tf.square(mu) - tf.exp(presig)), axis=1)
        kl_costs = tf.maximum(kl_costs, self.hps.kl_tolerance)
        return batch_z, kl_costs

    def config_model(self):
        # Input data configuration
        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[self.hps.batch_size], name="seq_lens")
        self.input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5], name="input_data")

        # data for use in conditional model encoder (only input or output??? TODO)
        # and decoder (input and output)
        self.input_x = self.input_data[:, :self.hps.max_seq_len, :]
        self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]

        # set the number of experts
        self.num_enc_experts = self.hps.num_enc_experts
        if not self.hps.conditional:
            self.num_enc_experts = 1

        # conditional mode, multiple experts
        self.input_xs = [self.input_x for _ in range(self.num_enc_experts)]

        # Decoder configuration
        if self.hps.dec_model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        # Dropout configuration and decoder
        # Decoder configuration
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True

        tf.logging.info('Input dropout mode = %s.', use_input_dropout)
        tf.logging.info('Output dropout mode = %s.', use_output_dropout)
        tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)

        cell = cell_fn(
            self.hps.dec_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        if use_input_dropout:
            tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                            self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                            self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell

        # either do vae-bit and get z, or do unconditional, decoder-only
        # Conditional generator and encoder
        # Encoder configuration
        if self.hps.enc_model == 'lstm':
            enc_cell_fn = rnn.LSTMCell
        elif self.hps.enc_model == 'layer_norm':
            enc_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.enc_model == 'hyper':
            enc_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        # define forward and backward function for bidirectional RNN
        # Encoder configuration
        self.enc_cell_fw = enc_cell_fn(
            self.hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            self.hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    def build_conditional_model(self):
        # config model detailed model elements
        # config input data
        self.config_model()

        # encode: compute the mu, presig for constructing latent vector z
        # the row of the output correspond to the number of experts
        self.experts_mu_presig = self.encoder(self.output_x, self.sequence_lengths, self.num_enc_experts)

        # TODO:should I save the params of the encoder?
        # wh it is not saved in check point?
        self.kl_weight = tf.get_variable(name="kl_weight", shape=[],
                                         initializer=tf.constant_initializer(self.hps.kl_weight_start), trainable=False)

        # reshape target data so that it is compatible with prediction shape
        # true data, not related to the model
        # cannot be used in inference mode, because we don't have enough x1_data, x2_data and pen_data
        target = tf.reshape(self.output_x, [-1, 5])
        self.x1_data, self.x2_data, eos_data, eoc_data, cont_data = tf.split(target, 5, 1)  # ???
        self.pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)  # the real pen state
        self.initial_states = []
        self.final_states = []
        self.gmm_outs = []
        self.batch_zs = []
        expert_kl_costs = []

        """Compute batch_z and kl_costs of each experts"""
        for i in range(self.num_enc_experts):
            mu, presig = self.experts_mu_presig[i]
            batch_z, kl_costs = self.compute_inputs(mu, presig)
            self.batch_zs.append(batch_z)
            expert_kl_costs.append(kl_costs)

        for i in range(self.num_enc_experts):
            # compute the hidden state for decoder
            batch_z = self.batch_zs[i]
            scope = "init_state%d" % i
            with tf.variable_scope(scope):
                initial_state = tf.nn.tanh(
                    rnn.super_linear(
                        batch_z,
                        self.cell.state_size,
                        init_w='gaussian',
                        weight_start=0.001,
                        input_size=self.hps.z_size))
            self.initial_states.append(initial_state)

        for i in range(self.num_enc_experts):
            # compute the input x for decoder
            batch_z = self.batch_zs[i]
            pre_tile_y = tf.reshape(batch_z, [self.hps.batch_size, 1, self.hps.z_size])  # ???
            overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])  # ???
            actual_input_x = tf.concat([self.input_xs[i], overlay_x], 2)

            # decoder: compute the output params for GMM and the final hidden state
            # final state: only useful for inference mode (which is the next state)
            initial_state = self.initial_states[i]
            out, last_state = self.decoder(actual_input_x, initial_state)
            self.gmm_outs.append(out)
            self.final_states.append(last_state)

            o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits = out

            # compute the reconstruction loss function for GMM
            r_costs = self.get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits,
                                        self.x1_data, self.x2_data, self.pen_data)
            r_costs = tf.reshape(r_costs, [self.hps.batch_size, -1])
            r_costs = tf.reduce_mean(r_costs, axis=1)

            # compute total costs
            kl_costs = expert_kl_costs[i]
            costs = kl_costs * self.kl_weight + r_costs

            # assign each sample to the expert with the lowest cost
            if i == 0:
                final_kl_costs = kl_costs
                final_r_costs = r_costs
                final_costs = costs
            else:
                idx = tf.less(costs, final_costs)
                final_kl_costs = tf.where(idx, kl_costs, final_kl_costs)
                final_r_costs = tf.where(idx, r_costs, final_r_costs)
                final_costs = tf.where(idx, costs, final_costs)

        # compute the average kl cost, reconstruction cost, total cost for the whole batch
        self.kl_cost = tf.reduce_mean(final_kl_costs)  # for printing only
        self.r_cost = tf.reduce_mean(final_r_costs)  # for printing only
        self.cost = tf.reduce_mean(final_costs)  # for printing only in validation mode, for optimizer in training mode

        if self.hps.is_training:
            # optimization: update all params in the computation graph
            self.global_step = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
                                               trainable=False)
            self.lr = tf.get_variable(name="lr", shape=[], initializer=tf.constant_initializer(self.hps.learning_rate),
                                      trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.cost)
            g = self.hps.grad_clip
            capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name="train_steps")

    def build_unconditional_model(self):
        # config model detailed model elements
        # config input data
        self.config_model()

        self.batch_z = tf.zeros(
            (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
        self.kl_cost = tf.zeros([], dtype=tf.float32)
        actual_input_x = self.input_x
        self.initial_state = self.cell.zero_state(
            batch_size=self.hps.batch_size, dtype=tf.float32)

        self.r_cost = self.decoder(actual_input_x, self.initial_state)

        if self.hps.is_training:
            self.cost = self.r_cost + self.kl_cost * self.hps.kl_weight

            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
            self.cost = self.r_cost + self.kl_cost * self.kl_weight

            gvs = optimizer.compute_gradients(self.cost)
            g = self.hps.grad_clip
            capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(
                capped_gvs, global_step=self.global_step, name='train_step')

    # the most important function in this file
    def build_model(self):
        """Define model architecture."""

        if self.hps.conditional:
            self.build_conditional_model()
        else:
            self.build_unconditional_model()


def sample(sess, model, seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
    """Samples a sequence from a pre-trained model."""

    def adjust_temp(pi_pdf, temp):
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """Samples from a pdf, optionally greedily."""
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_temp(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        tf.logging.info('Error with sampling ensemble.')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    if z is None:
        z = np.random.randn(1, model.hps.z_size)  # not used if unconditional

    if not model.hps.conditional:
        prev_state = sess.run(model.initial_state)
    else:  # in conditional mode, we have multiple experts
        feed = {l: r for l, r in zip(model.batch_zs, z)}
        prev_state = sess.run(model.initial_states, feed)

    # generate strokes by the best expert
    # (number of experts, seq_len, 5)
    # number of experts set to be 2
    strokes = np.zeros((model.num_enc_experts, seq_len, 5), dtype=np.float32)
    greedy = False
    temp = 1.0
    prev_xs = None  # store the previous output dot x

    for i in range(seq_len):
        if not model.hps.conditional:
            feed = {
                model.input_x: prev_x,
                model.sequence_lengths: [1],
                model.initial_state: prev_state
            }
        else:

            if prev_xs is None:
                prev_xs = [prev_x for _ in range(model.num_enc_experts)]
            feed = {
                l: r for l, r in zip(model.input_xs, prev_xs)
            }
            feed.update({
                model.sequence_lengths: [1]
            })
            feed.update({
                l: r for l, r in zip(model.initial_states, prev_state)
            })
            feed.update({
                l: r for l, r in zip(model.batch_zs, z)
            })

        # only output the results of the first state
        params = sess.run([
            model.gmm_outs, model.final_states
        ], feed)

        gmm_outs, final_states = params
        prev_state = []
        prev_xs = []

        for k in range(model.num_enc_experts):
            o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits = gmm_outs[k]
            next_state = final_states[k]

            if i < 0:
                greedy = False
                temp = 1.0
            else:
                greedy = greedy_mode
                temp = temperature

            idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

            idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
            eos = [0, 0, 0]
            eos[idx_eos] = 1

            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                                  o_sigma1[0][idx], o_sigma2[0][idx],
                                                  o_corr[0][idx], np.sqrt(temp), greedy)

            # compute reconstruction error for the best expert

            strokes[k, i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

            prev_x = np.zeros((1, 1, 5), dtype=np.float32)
            prev_x[0][0] = np.array(
                [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
            prev_xs.append(prev_x)
            prev_state.append(next_state)

    return strokes

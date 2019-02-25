from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from model_attn import GRUCellAttn, _linear
import util


def label_smooth(labels, num_class):  # 平滑标签
    labels = tf.one_hot(labels, depth=num_class)
    return 0.9 * labels + 0.1 / num_class


def get_optimizer(opt):  # 根据参数返回优化器  有adam和随机梯度下降
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn


class Model(object):
    def __init__(self, size, voc_size, num_layers, max_gradient_norm,
                 learning_rate, learning_rate_decay,
                 forward_only=False, optimizer="adam", decode="single"):
        self.voc_size = voc_size  # voc_size是什么
        self.size = size  # 大小
        self.num_layers = num_layers  # 层的数量
        self.learning_rate = learning_rate  #  学习率
        self.learning_decay = learning_rate_decay  # 指数下降的速率
        self.max_grad_norm = max_gradient_norm # 最大 随机 归一化？
        self.foward_only = forward_only # 是否只前向推导
        self.optimizer = optimizer # 优化器
        self.decode_method=decode # 解码的方法
        self.build_model() # 创建model

    def _add_place_holders(self):  # 添加holders的方法
        self.keep_prob = tf.placeholder(tf.float32)
        self.src_toks = tf.placeholder(tf.int32, shape=[None, None])  # tokens？ 这个是witness吗
        self.tgt_toks = tf.placeholder(tf.int32, shape=[None, None])  #
        self.src_mask = tf.placeholder(tf.int32, shape=[None, None])  # 这个应该是原始要训练的数据 填进这里来
        self.tgt_mask = tf.placeholder(tf.int32, shape=[None, None])  # target？ ??????????????????????????????
        self.beam_size = tf.placeholder(tf.int32)  # beam search的大小
        self.batch_size = tf.shape(self.src_mask)[1] # 每个 batch的大小
        self.len_inp = tf.shape(self.src_mask)[0]
        self.src_len = tf.cast(tf.reduce_sum(self.src_mask, axis=0), tf.int64)  # 原数据的长度
        # tf.cast 是类型转换函数 对原始数据按列求和，然后转换成int64的类型
        self.tgt_len = tf.cast(tf.reduce_sum(self.tgt_mask, axis=0), tf.int64)

    def setup_train(self):  # 训练计划
        self.lr = tf.Variable(float(self.learning_rate), trainable=False) # 不放在tranable里
        self.lr_decay_op = self.lr.assign(
            self.lr * self.learning_decay) # lr是什么
        self.global_step = tf.Variable(0, trainable=False) # 全局步骤
        params = tf.trainable_variables() # 参数
        opt = get_optimizer(self.optimizer)(self.lr)
        gradients = tf.gradients(self.losses, params) # 通过 loss函数和 params 进行梯度下降
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.max_grad_norm)
        # 该函数 保证梯度不爆炸或者消失
        self.gradient_norm = tf.global_norm(gradients) # 计算梯度额度全局范数
        self.param_norm = tf.global_norm(params) # 计算params的全局范数
        self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                           global_step=self.global_step) # 更新梯度

    def setup_embeddings(self):  # embeddings的设置
        with vs.variable_scope("embeddings"): # 在embeddings的作用域下
            zeros = tf.zeros([1, self.size]) # 1维 长度为size的0向量
            enc = tf.get_variable("L_enc", [self.voc_size - 1, self.size]) # encoder变量  #todo  ??????????
            self.L_enc = tf.concat([zeros, enc], axis=0) # 用来连接两个矩阵的操作，把zeros与encoder在列维度上进行拼接
            dec = tf.get_variable("L_dec", [self.voc_size - 1, self.size]) # decoder变量
            self.L_dec = tf.concat([zeros, dec], axis=0) # 拼接zeros与decoder
            # embedding encoder与decoder
            self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.src_toks)
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.tgt_toks)

    def setup_encoder(self): # encoder的设置
        with vs.variable_scope("Encoder"): # 在encoder的作用域下
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob) # 对encoder进行dropout dropout率为 keep_prob
            fw_cell = rnn_cell.GRUCell(self.size) # GRU单元
            fw_cell = rnn_cell.DropoutWrapper(
                fw_cell, output_keep_prob=self.keep_prob) # 对单元进行dropout
            self.encoder_fw_cell = rnn_cell.MultiRNNCell(  # 创建多层RNN的函数  encoder 前向单元
                [fw_cell] * self.num_layers, state_is_tuple=True)  # 设置multi-rnn cell
            bw_cell = rnn_cell.GRUCell(self.size)  # 设置size大小的GRU单元
            bw_cell = rnn_cell.DropoutWrapper(  # 根据dropout率，随机在抛弃GRU中计算的数据
                bw_cell, output_keep_prob=self.keep_prob)
            self.encoder_bw_cell = rnn_cell.MultiRNNCell(  # 设置 encoder 反向单元
                [bw_cell] * self.num_layers, state_is_tuple=True)
            out, _ = rnn.bidirectional_dynamic_rnn(self.encoder_fw_cell, # 设置动态双向RNN
                                                   self.encoder_bw_cell,
                                                   inp, self.src_len,
                                                   dtype=tf.float32,
                                                   time_major=True,
                                                   initial_state_fw=self.encoder_fw_cell.zero_state(
                                                       self.batch_size, dtype=tf.float32),  #  状态全部初始化为0
                                                   initial_state_bw=self.encoder_bw_cell.zero_state(
                                                       self.batch_size, dtype=tf.float32))
            out = tf.concat([out[0], out[1]], axis=2)  # 把 1 和 2拼接起来
            self.encoder_output = out

    def setup_decoder(self): # decoder的设置
        with vs.variable_scope("Decoder"):
            inp =  tf.nn.dropout(self.decoder_inputs, self.keep_prob)
            if self.num_layers > 1:
                with vs.variable_scope("RNN"):
                    decoder_cell = rnn_cell.GRUCell(self.size)
                    decoder_cell = rnn_cell.DropoutWrapper(decoder_cell,
                                                           output_keep_prob=self.keep_prob)
                    self.decoder_cell = rnn_cell.MultiRNNCell(
                        [decoder_cell] * (self.num_layers - 1), state_is_tuple=True)
                    inp, _ = rnn.dynamic_rnn(self.decoder_cell, inp, self.tgt_len,
                                             dtype=tf.float32, time_major=True,
                                             initial_state=self.decoder_cell.zero_state(
                                                 self.batch_size, dtype=tf.float32))

            with vs.variable_scope("Attn"):   #todo ???????????????????????????
                self.attn_cell = GRUCellAttn(self.size, self.len_inp,
                                             self.encoder_output, self.src_mask, self.decode_method)
                # 设置 attention
                self.decoder_output, _ = rnn.dynamic_rnn(self.attn_cell, inp, self.tgt_len,
                                                         dtype=tf.float32, time_major=True,
                                                         initial_state=self.attn_cell.zero_state(
                                                             self.batch_size, dtype=tf.float32,
                                                         ))
                # decoder的输出

    def setup_loss(self): # 损失函数的设置
        with vs.variable_scope("Loss"):
            len_out = tf.shape(self.decoder_output)[0]  # 输出的长度
            logits2d = _linear(tf.reshape(self.decoder_output,
                                                   [-1, self.size]),
                                        self.voc_size, True, 1.0) # 计算输出跟权重的矩阵相乘 #todo?????????
            self.outputs2d = tf.nn.log_softmax(logits2d)  # softmax
            targets_no_GO = tf.slice(self.tgt_toks, [1, 0], [-1, -1]) # 从第二行第一个开始裁剪，裁剪一个子矩阵出来，应该
            #  是把第一行裁剪掉的子矩阵
            masks_no_GO = tf.slice(self.tgt_mask, [1, 0], [-1, -1])
            # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
            labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1]) # 下面填充一行0  这个是标签
            if self.foward_only or self.keep_prob == 1.:
                labels1d = tf.one_hot(labels1d, depth=self.voc_size) # 如果只有前向，或者丢弃率是1，就one_hot编码
            else:
                labels1d = label_smooth(labels1d, self.voc_size)  # 否则对标签进行平滑
            mask1d = tf.reshape(tf.pad(masks_no_GO, [[0, 1], [0, 0]]), [-1])  # mask是什么~
            losses1d = tf.nn.softmax_cross_entropy_with_logits(logits=logits2d, labels=labels1d) * tf.to_float(mask1d)
            # 计算对数交叉熵
            losses2d = tf.reshape(losses1d, [len_out, self.batch_size])
            # 重构成 二维的
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(self.batch_size)
            # 最后reduce_sum

    def build_model(self): # 建模
        self._add_place_holders()   # 把所有的 holder都创建好
        with tf.variable_scope("Model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_encoder()
            self.setup_decoder()
            self.setup_loss()
            if self.foward_only:
                self.setup_beam()
        if not self.foward_only:
            self.setup_train()
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def decode_step(self, inputs, state_inputs): # decode步骤
        beam_size = tf.shape(inputs)[0]
        with vs.variable_scope("Decoder", reuse=True):
            with vs.variable_scope("RNN", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    rnn_out, rnn_outputs = self.decoder_cell(inputs, state_inputs[:self.num_layers-1])
            with vs.variable_scope("Attn", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    if self.decode_method == 'average':
                        out, attn_outputs = self.attn_cell.beam_average(rnn_out, state_inputs[-1], beam_size)
                    elif self.decode_method == 'weight':
                        out, attn_outputs = self.attn_cell.beam_weighted(rnn_out, state_inputs[-1], beam_size)
                    elif self.decode_method == 'flat':
                        out, attn_outputs = self.attn_cell.beam_flat(rnn_out, state_inputs[-1], beam_size)
                    else:
                        out, attn_outputs = self.attn_cell.beam_single(rnn_out, state_inputs[-1], beam_size)
        state_outputs = rnn_outputs + (attn_outputs, )
        return out, state_outputs

    def setup_beam(self): # 设置 beam search
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[util.SOS_ID]])
        beam_probs_0 = tf.constant([0.])
        cand_seqs_0 = tf.constant([[util.EOS_ID]])
        cand_probs_0 = tf.constant([-3e38])

        state_0 = tf.zeros([1, self.size])
        states_0 = [state_0] * self.num_layers

        def beam_cond(cand_probs, cand_seqs, time, beam_probs, beam_seqs, *states):
            return tf.logical_and(tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs),
                                  time < tf.reshape(self.len_inp, ()) + 10)

        def beam_step(cand_probs, cand_seqs, time, beam_probs, beam_seqs, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
            decoder_input = embedding_ops.embedding_lookup(self.L_dec, inputs)
            decoder_output, state_output = self.decode_step(decoder_input, states)

            with vs.variable_scope("Loss", reuse=True):
                do2d = tf.reshape(decoder_output, [-1, self.size])
                logits2d = _linear(do2d, self.voc_size, True, 1.0)
                logprobs2d = tf.nn.log_softmax(logits2d)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [batch_size, util.EOS_ID]),
                                              tf.tile([[-3e38]], [batch_size, 1]),
                                              tf.slice(total_probs, [0, util.EOS_ID + 1],
                                                       [batch_size, self.voc_size - util.EOS_ID - 1])],
                                          axis=1)
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])

            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.voc_size)
            next_mods = tf.mod(top_indices, self.voc_size)

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases),
                                           tf.reshape(next_mods, [-1, 1])], axis=1)

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], axis=0)
            EOS_probs = tf.slice(total_probs, [0, util.EOS_ID], [batch_size, 1])

            new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], axis=0)
            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)


            return [next_cand_probs, next_cand_seqs, time + 1, next_beam_probs, next_beam_seqs] + next_states

        var_shape = []
        var_shape.append((cand_probs_0, tf.TensorShape([None, ])))
        var_shape.append((cand_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_probs_0, tf.TensorShape([None, ])))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(*var_shape)
        self.loop_vars = loop_vars
        self.loop_var_shapes = loop_var_shapes
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, shape_invariants=loop_var_shapes, back_prop=False)
        self.vars = ret_vars
        self.beam_output = ret_vars[1]
        self.beam_scores = ret_vars[0]

    def decode_beam(self, session, encoder_output, src_mask, len_inp, beam_size=128):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.src_mask] = src_mask
        input_feed[self.len_inp] = len_inp
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size
        output_feed = [self.beam_output, self.beam_scores]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def encode(self, session, src_toks, src_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.encoder_output]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def train(self, session, src_toks, src_mask, tgt_toks, tgt_mask, dropout):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_mask] = tgt_mask
        input_feed[self.keep_prob] = 1 - dropout
        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm]
        outputs = session.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3]

    def test(self, session, src_toks, src_mask, tgt_toks, tgt_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_mask] = tgt_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

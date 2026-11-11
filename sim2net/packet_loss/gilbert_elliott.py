#!/usr/bin/env python
# -*- coding: utf-8 -*-


# (c) 2012 Michal Kalewski  <mkalewski at cs.put.poznan.pl>
#
# This file is a part of the Simple Network Simulator (sim2net) project.
# USE, MODIFICATION, COPYING AND DISTRIBUTION OF THIS SOFTWARE IS SUBJECT TO
# THE TERMS AND CONDITIONS OF THE MIT LICENSE.  YOU SHOULD HAVE RECEIVED A COPY
# OF THE MIT LICENSE ALONG WITH THIS SOFTWARE; IF NOT, YOU CAN DOWNLOAD A COPY
# FROM HTTP://WWW.OPENSOURCE.ORG/.
#
# For bug reports, feature and support requests please visit
# <https://github.com/mkalewski/sim2net/issues>.

"""
This module provides an implementation of the Gilbert-Elliott packet loss
model.

The Gilbert-Elliott model ([Gil60]_, [Ell63]_) describes error patterns in
communication channels ([HH08]_).  The model is based on a simple *Markov
chain* with two states: ``G`` (for `good` or `gap`) and ``B`` (for `bad` or
`burst`).  Each of them may generate errors (packet losses) as independent
events at a state dependent error rate: :
math:`1-k` in the `good` state and
:math:`1-h` in the `bad` state.  The chain is shown in the figure below along
with the transition matrix :math:`A` that uses two transitions:
:math:`p=P(q_t=B|q_{t-1}=G)` and :math:`r=P(q_t=G|q_{t-1}=B)` (:math:`q_t`
denotes the state at time :math:`t`)::
r是由状态B转移到G的概率，p是由状态G转移到B的概率，(1−k)是在状态G的丢包概率，(1−h)是在状态B的丢包概率。

             +-------+      p      +-------+                   {          }
        +----|       |------------>|       |<---+              { 1-p   p  }
    1-p |    |   G   |             |   B   |    | 1-r      A = {          }
        |    | (1-k) |             | (1-h) |    |              {  r   1-r }
        +--->|       |<------------|       |----+              {          }
             +-------+      r      +-------+

Then, error rate :math:`p_E` is obtained (in steady mode) for the model as
follows: :math:`p_E=(1-k)\times\frac{r}{p+r}+(1-h)\times\frac{p}{p+r}`
(assuming: :math:`0<p,r<1`).
采用GE模型 长期情况下的丢包率：(1-k) * (r / (p+r))+(1-h) * (p / (p+r))
[p, r, h, k] to [0.378, 0.883, 0.810, 0.938]
0.04341475+0.05695479=0.10036954

It is worth to note that when :math:`q=1-p` (and :math:`k=1, h=0`), this model
reduces to the Bernoulli model -- a very simple loss model, characterized by a
single parameter, the loss rate :math:`r`, used for modeling packet loss.
:math:`k=1, h=0`退化为伯努利 好网络一定不出错不丢包 可以用来建模每个时刻的网络的绝对好坏状态 然后直接根据绝对的k和h判断这个时刻一定丢包和不丢包
Finally, :math:`p` equal to :math:`0` means that no losses are possible,
whereas :math:`r` equal to :math:`0` means that no transmission is successful
(once the ``B`` state is reached).


.. [Ell63]  E. O. Elliott.  Estimates of Error Rates for Codes on Burst-Noise
   Channels.  In *Bell System Technical Journal*, vol. 42(5), 1977--1997.  Bell
   Laboratories, September 1963.
.. [Gil60]  Edgar Nelson Gilbert.  Capacity of a Burst-Noise Channel.  In *Bell
   System Technical Journal*, vol. 39(5), 1253--1265.  Bell Laboratories,
   September 1960.
.. [HH08]  Gerhard Haßlinger, Oliver Hohlfeld.  The Gilbert-Elliott Model for
   Packet Loss in Real Time Services on the Internet.  In  Proceedings of the
   *14th GI/ITG Conference on Measurement, Modelling and Evaluation of Computer
   and Communication Systems* (MMB 2008), pp. 269--286.  Dortmund, Germany,
   April 2008.
"""

import os
import sys
sys.path.append(os.getcwd())
from sim2net.packet_loss._packet_loss import PacketLoss #基类  # pyright: ignore[reportMissingImports]
from sim2net.utility.validation import check_argument_type #参数检查工具 `check_argument_type`  # pyright: ignore[reportMissingImports]


__docformat__ = 'reStructuredText'

#可以输入一个包含prhk的元祖，每运行一次packet_loss函数会预测当前时间步的好坏状态，然后并返回当前时间步骤是否丢包。true为丢包
#创建了一个sim2net.packet_loss.packetloss的logger对象 记录类型检测异常但只会打印到终端上
#pylint: disable=C0103
class GilbertElliott(PacketLoss):
    """
    This class implements the Gilbert-Elliott packet loss model.
    """

    #: Default value of the `p` parameter.
    __DEFAULT_P = 0.00001333

    #: Default value of the `r` parameter.
    __DEFAULT_R = 0.00601795

    #: Default value of the `h` parameter.
    __DEFAULT_H = 0.55494900

    #: Default value of the `k` parameter.
    __DEFAULT_K = 0.99999900

    #prhk 是一个可选参数，它应该是一个包含 p, r, h, k 四个值的元组 (tuple)
    def __init__(self, prhk=None):
        """
        *Parameters*:
            - **prhk** (`tuple`): a `tuple` that contains four model
              parameters: :math:`0\leqslant p,r,h,k\leqslant 1`, respectively
              (each of type `float`).  The parameters default to the following
              values:

              * :math:`p=0.00001333`,
              * :math:`r=0.00601795`,
              * :math:`h=0.55494900`,
              * :math:`k=0.99999900`;

              (which leads to error rate equal to :math:`0.098\\%` and the mean
              packet loss rate equal to :math:`0.1\\%` ([HH08]_)).导致（理论）错误率等于 0.098%，以及平均丢包率等于 0.1%

        *Raises*:
            - **ValueError**: raised when the given value any model parameter
              is less than zero or greater that one.

        (At the beginning the model is in the ``G`` state.)
        """
        super(GilbertElliott, self).__init__(self.__class__.__name__)#调用父类 PacketLoss 的构造函数
        if prhk is None:
            p = float(GilbertElliott.__DEFAULT_P)
            r = float(GilbertElliott.__DEFAULT_R)
            b = float(1.0 - GilbertElliott.__DEFAULT_H) #在坏状态下失败 (丢包) 的概率。
            g = float(1.0 - GilbertElliott.__DEFAULT_K) #在好状态下失败 (丢包) 的概率。
        else:
            for param in range(4):
                #使用 sim2net 的工具函数来验证每个参数prhk[param]是不是 float 类型
                check_argument_type(GilbertElliott.__name__, #__name__返回该对象的名字（以字符串的形式）
                                    'prhk[' + str(param) + ']', float,
                                    prhk[param], self.logger)
                if prhk[param] < 0.0 or prhk[param] > 1.0: #检查范围
                    raise ValueError('Parameter "prhk[%d]": a value of the' \
                                     ' model parameter cannot be less than' \
                                     ' zero and greater than one but %f given!'
                                     % (param, float(prhk[param])))
            p = float(prhk[0])
            r = float(prhk[1])
            b = float(1.0 - prhk[2])
            g = float(1.0 - prhk[3])
        # ( current state: 'G' or 'B',
        #   transition probability,
        #   current packet error rate )
        self.__state_g = ('G', p, g)
        self.__state_b = ('B', r, b)
        self.__current_state = self.__state_g

    def packet_loss(self):
        """
        Returns information about whether a transmitted packet has been lost or
        can be successfully received by destination node(s) according to the
        Gilbert-Elliott packet loss model.

        *Returns*:
            (`bool`) `True` if the packet has been lost, or `False` otherwise.
        """
        transition = self.random_generator.uniform(0.0, 1.0)
        if transition <= self.__current_state[1]:
            if self.__current_state[0] == 'G':
                self.__current_state = self.__state_b
            else:
                self.__current_state = self.__state_g
        loss = self.random_generator.uniform(0.0, 1.0)
        if loss <= self.__current_state[2]:
            return True
        return False

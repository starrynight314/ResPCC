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
Provides a pseudo-random number generator.
"""


import random


__docformat__ = 'reStructuredText'

#: A random generator object of the :class:`_Randomness` class -- to create the
#: object a call to the :func:`get_random_generator` function is necessary.
__RANDOM_GENERATOR = None


class _Randomness(object):
    """
    This class provides a pseudo-random number generator with the use of the
    :mod:`random` module from the standard library that produces a sequence of
    numbers that meet certain statistical requirements for randomness.
    """

    def __init__(self):
        self.__random = random.Random() #创建了 random 模块中 Random 类的一个实例
    #用一个外部提供的“状态对象”来重置或恢复内部随机数生成器的状态。
    def set_state(self, generator_state):
        """
        Sets a new internal state of the generator.

        The state can be obtained from a call to :meth:`get_state` method.

        *Parameters*:
            - **generator_state**: an internal state of the generator to set.

        *Raises*:
            - **ValueError**: raised when a given value of the
              *generator_state* parameter is `None`.
        """
        if generator_state is None:
            raise ValueError('Parameter "generator_state": a random' \
                             ' generator state object expected but "None"' \
                             ' value given!')
        self.__random.setstate(generator_state)
    #捕获并返回内部随机数生成器当前这一刻的内部状态。
    #可以随时调用 get_state() 来“存档”，然后可以在未来的任何时候，通过 set_state() 来“读档”，让随机数生成器精确地回到你存档时的那个点。
    def get_state(self):
        """
        Returns an object capturing the current internal state of the
        generator.

        This object can be passed to :meth:`set_state` to restore the state.
        """
        return self.__random.getstate()
    #生成一个 [begin, end] 范围内的均匀分布的随机浮点数。
    def uniform(self, begin, end):
        """
        Returns a random floating point number :math:`N` such that
        :math:`begin\\leqslant N\\leqslant end` for :math:`begin\\leqslant end`
        and :math:`end\\leqslant N\\leqslant begin` for :math:`end < begin`.
        """
        return self.__random.uniform(begin, end)
    #符合正态（高斯）分布的随机浮点数
    def normal(self, mikro, sigma):
        """
        Returns a random floating point number with the normal (i.e. Gaussian)
        distribution.

        *Parameters*:
            - **mikro** (`float`): a value of the mean to be used by the
              generator;
            - **sigma** (`float`): a value of the standard deviation to be used
              by the generator.
        """
        return self.__random.gauss(mikro, sigma)
    # 原地 (in-place) 打乱一个序列（比如一个列表 list）的元素顺序。
    #这个方法没有返回值。它直接修改了你传入的那个 sequence 对象本身。
    def random_order(self, sequence):
        """
        Shuffles the given **sequence** *in place*.
        """
        self.__random.shuffle(sequence)

#如果没有创建随机数对象创建一个 所以全局只会创建一次随机数对象 创建之后 所有子类获得的随机数对象是一样的
def get_random_generator():
    """
    Returns an object representing the :class:`_Randomness` pseudo-random
    number generator.  Multiple calls to this function will return the same
    object.
    """
    global __RANDOM_GENERATOR
    if __RANDOM_GENERATOR is None:
        __RANDOM_GENERATOR = _Randomness()
    return __RANDOM_GENERATOR

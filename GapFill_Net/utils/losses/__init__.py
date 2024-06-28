# -*- coding: utf-8 -*-
# @Author: Chang-jiang.Shi
# @Date:   2022-03-28 10:46:19
# @Last Modified by:   Chang-jiang.Shi
# @Last Modified time: 2022-03-29 15:01:55
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss, WeightedDiceLoss
from .focal import FocalLoss
from .jaccard import JaccardLoss
from .lovasz import LovaszLoss
from .soft_bce import SoftBCEWithLogitsLoss
from .soft_ce import SoftCrossEntropyLoss
from .tversky import TverskyLoss

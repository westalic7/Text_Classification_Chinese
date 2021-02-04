# -*- coding:utf-8 -*-

from model.AVCNN import AVCNN_Model
from model.AVRNN import AVRNN_Model
from model.BIGRU import BIGRU_Model
from model.BILSTM import BILSTM_Model
from model.BILSTM_Attention import BILSTM_Attention_Model
from model.CNN import CNN_Model
from model.CNN_GRU import CNN_GRU_Model
from model.DPCNN import DPCNN_Model
from model.FastText import FastText_Model
from model.R_CNN import R_CNN_Model
from model.TEXTCNN import TEXTCNN_Model
from model.BaseEmbedding import BaseEmbedding_Model

MODELS_FILTER = {
    'avcnn': AVCNN_Model,
    'avrnn': AVRNN_Model,
    'embed':BaseEmbedding_Model,
    'bigru': BIGRU_Model,
    'bilstm': BILSTM_Model,
    'bilstm_attn': BILSTM_Attention_Model,
    'cnn': CNN_Model,
    'cnngru': CNN_GRU_Model,
    'dpcnn': DPCNN_Model,
    'fasttext': FastText_Model,
    'rcnn': R_CNN_Model,
    'textcnn': TEXTCNN_Model
}

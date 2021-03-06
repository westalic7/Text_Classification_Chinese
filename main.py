# -*- coding:utf-8 -*-
import time
import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from model import MODELS_FILTER
from utils.preprocess import read_data_file, BertClassificationProcessor, ClassificationProcessor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data.dataloader import DataLoader
from utils.dataset import ClassificationDataSet
from utils.logger import LoggerClass

rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

Processor = {
    'bertVocab': BertClassificationProcessor,
    'wordCount': ClassificationProcessor
}


class DefaultConfig:
    def __init__(self):
        # debug mode
        self.DEBUG = False
        self.data_debug_samples_rate = 0.1 if self.DEBUG else 1

        # train parameters
        self.random_seed = 1234
        self.data_path = './data/cnews/'
        self.processor_type = 'bertVocab'
        self.model_name = 'cnn'
        self.seq_len = 100
        self.embd_len = 100
        self.batch_size = 256
        self.train_epochs = 40
        self.lr = 1E-3
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # log
        self.logger = LoggerClass('log', logger_file='./checkpoint/train_{}_{}_{}_{}.log'.format(
            self.model_name,
            self.seq_len,
            self.embd_len,
            int(time.time())
        ))
        self.logger.info(f'Model name: {self.model_name}')
        self.logger.info(f'Data dir: {self.data_path}')


class TextClassification(DefaultConfig):

    def __init__(self):
        super(TextClassification, self).__init__()
        self.model_func = MODELS_FILTER[self.model_name]
        self.load_embedding_only = False  # 只加载模型Embedding层
        self.train_embedding = True  # 是否训练Embedding层
        # self.pretrain_model_path = f'./checkpoint/fc8765.checkpoint_BaseEmbedding_Model.pt'  # 预训练Embedding模型地址，
        self.pretrain_model_path = f'./checkpoint/checkpoint_{self.model_func.__name__}.pt'
        self.save_model_path = f'./checkpoint/checkpoint_{self.model_func.__name__}.pt'
        self.clfp = Processor[self.processor_type](sequence_length=self.seq_len,
                                                   vocab_path='./utils/vocab.txt')  # 数据处理过程定义seq_len，模型中不需要再定义

        self.set_seed(self.random_seed)
        self.prepare_dataset()
        self.prepare_model()
        self.freeze_layers()
        self.prepare_optimizer()

    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def prepare_dataset(self):
        """
        加载并缓存数据集，数据预处理
        """
        cached_train_file = os.path.join(self.data_path, 'cached_cls-{}-{}-{}-{}'.format(
            self.data_path.split('/')[-2], self.processor_type, self.seq_len, 'train'))
        cached_test_file = os.path.join(self.data_path, 'cached_cls-{}-{}-{}-{}'.format(
            self.data_path.split('/')[-2], self.processor_type, self.seq_len, 'test'))
        cached_token2idx_file = os.path.join(self.data_path, 'cached_cls-{}-{}-{}-{}'.format(
            self.data_path.split('/')[-2], self.processor_type, self.seq_len, 'token2idx'))
        if os.path.exists(cached_train_file) and os.path.exists(cached_test_file):
            self.logger.info("Loading datasets from cached file %s", cached_train_file)
            _train_dataset = torch.load(cached_train_file)
            self.logger.info("Loading datasets from cached file %s", cached_test_file)
            _test_dataset = torch.load(cached_test_file)
            self.logger.info("Loading token2idx from cached file %s", cached_token2idx_file)
            self.clfp.token2idx = torch.load(cached_token2idx_file)
        else:
            self.logger.info("Loading datasets from datasets file.")
            train_x, train_y = read_data_file(os.path.join(self.data_path, 'cnews.{}.txt'.format('train')),
                                              self.data_debug_samples_rate)
            test_x, test_y = read_data_file(os.path.join(self.data_path, 'cnews.{}.txt'.format('test')),
                                            self.data_debug_samples_rate)
            self.clfp.analyze_corpus(train_x + test_x, train_y + test_y)
            self.logger.info(f'Analyze corpus, token2idx get token nums: {len(self.clfp.token2idx)}')
            self.logger.info("Saving token2idx into cached file %s", cached_token2idx_file)
            torch.save(self.clfp.token2idx, cached_token2idx_file)

            def save_cached(dataset_x, dataset_y, datatype='train'):
                dataset_x, dataset_y = self.clfp.process_x_dataset(dataset_x), self.clfp.process_y_dataset(dataset_y)
                _dataset = ClassificationDataSet(dataset_x, dataset_y)
                del dataset_x, dataset_y

                if datatype == 'train':
                    self.logger.info("Saving datasets into cached file %s", cached_train_file)
                    torch.save(_dataset, cached_train_file)
                else:
                    self.logger.info("Saving datasets into cached file %s", cached_test_file)
                    torch.save(_dataset, cached_test_file)

                return _dataset

            _train_dataset = save_cached(train_x, train_y, 'train')
            _test_dataset = save_cached(test_x, test_y, 'test')

        self.token2idx = self.clfp.token2idx
        self.train_data_loader = DataLoader(dataset=_train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        self.valid_data_loader = DataLoader(dataset=_test_dataset,
                                            batch_size=self.batch_size)

    def prepare_model(self):
        """
        加载模型，或只加载Embedding层权重
        """
        # 判断设备类型gpu或cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.backends.cudnn.deterministic = True if self.device == 'cuda' else False
        self.logger.info(f'Device: {self.device}')
        # 加载模型
        self._model = self.model_func(seq_len=self.seq_len,
                                      embed_len=self.embd_len,
                                      token2idx_len=len(self.clfp.token2idx))

        if os.path.exists(self.pretrain_model_path):
            # self._model.load_state_dict(torch.load(self.pretrain_model_path,
            #                                        map_location=self._map_location,))
            model_dict_training = self._model.state_dict()
            model_dict_pretrained = torch.load(self.pretrain_model_path, map_location=self.device, )
            if self.load_embedding_only:
                pretrained_dict = {k: v for k, v in model_dict_pretrained.items()
                                   if k in model_dict_training and k == 'embedding_layer.weight'}
            else:
                pretrained_dict = {k: v for k, v in model_dict_pretrained.items() if k in model_dict_training}

            self.logger.info(f'Pretrain model: {self.pretrain_model_path}')
            self.logger.info('Checkpoint found, continue training!')
            self.logger.info('Embedding layer info: {}'.format(pretrained_dict['embedding_layer.weight'].shape))
            # 更新现有的model_dict
            model_dict_training.update(pretrained_dict)
            # 加载我们真正需要的state_dict
            self._model.load_state_dict(model_dict_training)
        else:
            self.logger.info('Checkpoint not found, training from initial!')

        self._model.to(self.device)
        self.logger.info(self._model)

    def freeze_layers(self):
        """
         冻结参数（可选）
         """
        # 冻结不需要训练的层
        for param in self._model.parameters():
            param.requires_grad = True
        for param in self._model.embedding_layer.parameters():
            param.requires_grad = self.train_embedding
        params_info = get_parameter_number(self._model)
        self.logger.info(f'Parameters info: {params_info}')

    def prepare_optimizer(self):
        """
        选择优化器
        # optimizer = torch.optim.RMSprop(self._model.parameters(), alpha=0.9)
        # optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=0.8)
        # optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, betas=(0.9,0.99))
        # optimizer = torch.optim.ASGD(self._model.parameters(), lr=lr)
        # optimizer = torch.optim.Adagrad(self._model.parameters(), lr=lr)
        # optimizer = torch.optim.Adadelta(self._model.parameters())
        # optimizer = torch.optim.Adamax(self._model.parameters())
        # optimizer = torch.optim.AdamW(self._model.parameters())
        # optimizer = torch.optim.Rprop(self._model.parameters())
        """
        # self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self._model.parameters()), alpha=0.9)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=self.lr,
                                          betas=(0.9, 0.99))
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self._model.parameters()), lr=self.lr, momentum=0.8)

    def train(self):

        best_metrics_score = 0
        for epc in range(self.train_epochs):
            self._model.train()
            total_pred_train = []
            total_true_train = []
            total_loss = 0.
            total_num = 0.

            for batch_idx, batch_data in tqdm(enumerate(self.train_data_loader), desc=f'Train:: Epoch {epc}:'):
                text_index, target = batch_data
                text_index = text_index.to(self.device)
                target = target.to(self.device)
                y_pred = self._model(text_index)

                # batch loss
                loss = self.loss_fn(y_pred, target.long())
                # total loss
                total_loss += float(loss)
                total_num += target.shape[0]
                avgloss = total_loss / total_num

                # labels
                predict_label = y_pred.argmax(1).float()
                true_label = target.float()
                total_pred_train.extend(predict_label.tolist())
                total_true_train.extend(true_label.tolist())

                # 反向传播和参数更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # # lr更新策略
            # lr = lr*0.95

            # 训练集metrics
            _, _, train_f1, _ = precision_recall_fscore_support(total_true_train,
                                                                total_pred_train,
                                                                labels=list(set(total_true_train)),
                                                                average='weighted')
            train_acc = accuracy_score(total_true_train, total_pred_train)
            train_loss_avg = avgloss

            # 验证集metrics
            valid_f1, valid_acc, valid_loss_avg = self.validation()

            metrics_score = valid_acc
            if metrics_score > best_metrics_score:
                best_metrics_score = metrics_score
                torch.save(self._model.state_dict(), self.save_model_path)
                update_flag = '*'
            else:
                update_flag = ' '

            self.logger.info("| epoch: {:2d} "
                             "| - train -| loss: {:4.4f}| f1: {:.4f} | acc: {:.4f} "
                             "| - valid -| loss: {:4.4f}| f1: {:.4f} | acc: {:.4f} | {}".format(epc + 1,
                                                                                                train_loss_avg,
                                                                                                train_f1,
                                                                                                train_acc,
                                                                                                valid_loss_avg,
                                                                                                valid_f1,
                                                                                                valid_acc,
                                                                                                update_flag))

    def validation(self, ):
        """
        计算验证集metrics
        """
        self._model.eval()

        total_pred = []
        total_true = []
        total_loss = 0.
        total_num = 0.
        for batch_idx, batch_data in tqdm(enumerate(self.valid_data_loader), desc='Valid:'):
            text_index, target = batch_data
            text_index = text_index.to(self.device)
            target = target.to(self.device)
            a = target.shape[0]

            y_pred = self._model(text_index)
            # loss
            loss = self.loss_fn(y_pred, target.long())
            total_loss += float(loss)
            total_num += a
            avgloss = total_loss / total_num
            predict_label = y_pred.argmax(1).float()
            true_label = target.float()

            total_pred.extend(predict_label.tolist())
            total_true.extend(true_label.tolist())

        _, _, f1, _ = precision_recall_fscore_support(total_true,
                                                      total_pred,
                                                      labels=list(set(total_true)),
                                                      average='weighted')
        acc = accuracy_score(total_true, total_pred)
        return f1, acc, avgloss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    tcls = TextClassification()
    tcls.train()

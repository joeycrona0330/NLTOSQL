import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from models.embedding import EmbeddingLayer
from models.predictors import AggregatePredictor, SelectPredictor, ConditionPredictor
import constants.main_constants as const
import services.torch as torch_services
from services.logger import Logger


class NLQModel(nn.Module):
    def __init__(self, args, token_to_index, token_weights):
        super(NLQModel, self).__init__()
        self.args = args
        self.token_to_index = token_to_index

        self.aggregate_embedding_layer = EmbeddingLayer(emb_size=args.emb_size, gpu=args.gpu, token_to_index=token_to_index, token_weights=token_weights)
        # self.select_embedding_layer = EmbeddingLayer(emb_size=args.emb_size, gpu=args.gpu, token_to_index=token_to_index, token_weights=token_weights)
        # self.condition_embedding_layer = EmbeddingLayer(emb_size=args.emb_size, gpu=args.gpu, token_to_index=token_to_index, token_weights=token_weights)

        self.aggregate_predictor = AggregatePredictor(embedding_layer=self.aggregate_embedding_layer, args=args)
        # self.select_predictor = SelectPredictor(embedding_layer=self.select_embedding_layer, args=args)
        # self.condition_predictor = ConditionPredictor(embedding_layer=self.condition_embedding_layer, args=args)

        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if args.gpu:
            self.cuda()

    def forward(self, input):
        input = Variable(input.long())
        if self.args.gpu:
            input = input.cuda()

        return self.aggregate_predictor.forward(input)

    def accuracy(self, query_model, sql_model):
        true_set = np.array([])
        predicted_set = np.array([])
        for b, (input, true_output) in enumerate(zip(query_model, sql_model)):
            self.aggregate_predictor.reset_hidden_state()
            true_output = Variable(true_output.long())
            if self.args.gpu:
                true_output = true_output.cuda()
            logits = self.forward(input)
            logits = torch_services.get_numpy(logits, self.args.gpu)
            predicted_output = np.argmax(logits, 1)
            predicted_set = np.append(predicted_set, predicted_output)
            true_set = torch_services.append(true_set, true_output, self.args.gpu)
        return 100 * torch_services.accuracy_score(true_set, predicted_set)

    def start_train(self, optimizer, train_query_model, train_sql_model, dev_query_model, dev_sql_model):
        logger = Logger()
        num_batches = len(train_query_model)
        total_batches = self.args.epochs * num_batches
        best_val_accuracy = 0
        try:
            for e in range(self.args.epochs):
                logger.start_timer('Epoch %d training..' % (e + 1))
                for b, (input, true_output) in enumerate(zip(train_query_model, train_sql_model)):
                    optimizer.zero_grad()
                    self.aggregate_predictor.reset_hidden_state()
                    true_output = Variable(true_output.long())
                    if self.args.gpu:
                        true_output = true_output.cuda()
                    logits = self.forward(input)
                    loss = self.cross_entropy_loss(logits, true_output)
                    loss.backward()
                    optimizer.step()
                    clip_grad_norm(self.parameters(), const.AGG_GRAD_CLIP)
                    loss = torch_services.get_numpy(loss, self.args.gpu)[0]
                    accuracy = 100 * torch_services.accuracy(true_output, logits)
                    print('Batch [{:d}/{:d}] | Epoch {:d} | Loss: {:.3f} | Accuracy: {:.2f}%'.format((b + 1) * (e + 1), total_batches, e + 1, loss, accuracy))

                logger.end_timer('Epoch {:d}'.format(e + 1))
                print('Calculating Accuracy..')
                train_accuracy = self.accuracy(train_query_model, train_sql_model)
                val_accuracy = self.accuracy(dev_query_model, dev_sql_model)
                print('Train Accuracy: {:.2f}% | Validation Accuracy: {:.2f}%'.format(train_accuracy, val_accuracy))
                if val_accuracy > best_val_accuracy:
                    logger.start_timer('Saving the best model..')
                    torch.save(self.aggregate_embedding_layer.state_dict(), const.AGG_EMB_SAVE_MODEL.format(val_accuracy))
                    torch.save(self.aggregate_predictor.state_dict(), const.AGG_SAVE_MODEL.format(val_accuracy))
                    best_val_accuracy = val_accuracy
                    logger.end_timer('Save Complete!')
        except KeyboardInterrupt:
            print('-' * 55)
            print('Exiting from training..')

# TODO: Make all Predictors, forward function and loss.

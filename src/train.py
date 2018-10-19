from argparse import ArgumentParser

import torch

from models.data import QueryDataModel, SQLDataModel
from services.common import load_data, make_token_to_index
from services.logger import Logger
from models.glove import Glove
from models.model import NLQModel
import constants.main_constants as const

parser = ArgumentParser()
parser.add_argument('--gpu', action='store_true',
                    help='Use GPU')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch Size')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate')
parser.add_argument('--decay', type=float, default=0.95,
                    help='Decay for Learning Rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of Epochs')
parser.add_argument('--save', default='save',
                    help='Model save directory.')
parser.add_argument('--debug', action='store_true',
                    help='Fast debugging mode.')
parser.add_argument('--hard_reload', action='store_true',
                    help='Pre-processing will be done from scratch.')
args = parser.parse_args()
logger = Logger()

# TODO: Make a components check function in services.common to check for data, glove and directories.

logger.start_timer('Loading data..')
train_query_list, train_sql_list, train_table_data, train_db = load_data(data_dir=const.DATA_DIR, split='train', debug=args.debug)
dev_query_list, dev_sql_list, dev_table_data, dev_db = load_data(data_dir=const.DATA_DIR, split='dev', debug=args.debug)
test_query_list, test_sql_list, test_table_data, test_db = load_data(data_dir=const.DATA_DIR, split='test', debug=args.debug)
logger.end_timer()

glove = Glove(file_name=const.GLOVE, load_if_exists=(True and not args.hard_reload))
args.emb_size = glove.length

logger.start_timer('Making token dictionary..')
token_to_index, token_weights = make_token_to_index(data=train_query_list, embedding=glove, use_extra_tokens=True, load_if_exists=(True and not args.hard_reload))
logger.end_timer()

logger.start_timer('Making data models..')
train_query_model = QueryDataModel(query_list=train_query_list, token_to_index=token_to_index, batch_size=args.batch_size)
train_sql_model = SQLDataModel(sql_list=train_sql_list, batch_size=args.batch_size)
dev_query_model = QueryDataModel(query_list=dev_query_list, token_to_index=token_to_index, batch_size=args.batch_size)
dev_sql_model = SQLDataModel(sql_list=dev_sql_list, batch_size=args.batch_size)
logger.end_timer()

nlq_model = NLQModel(args=args, token_to_index=token_to_index, token_weights=token_weights)
optimizer = torch.optim.Adam(nlq_model.parameters(), lr=args.lr, weight_decay=args.decay)
nlq_model.start_train(optimizer, train_query_model, train_sql_model, dev_query_model, dev_sql_model)

import numpy as np
import pandas as pd
import argparse
from os import path
from datetime import datetime as dt

from hgru4rec import HGRU4Rec as RNN
from evaluation import evaluate_sessions_batch_hier_bootstrap
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lastfm')
parser.add_argument('--split', type=str,default='lso-valid')
parser.add_argument('--session_layers', type=str, default='1')
parser.add_argument('--user_layers', type=str, default='1')
parser.add_argument('--loss', type=str, default='bpr-max')
parser.add_argument('--hidden_act', type=str, default='tanh')
parser.add_argument('--adapt', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.1)
parser.add_argument('--dropout_p_hidden_usr', type=float, default=0.0)
parser.add_argument('--dropout_p_hidden_ses', type=float, default=0.0)
parser.add_argument('--dropout_p_init', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=0.0)
parser.add_argument('--grad_cap', type=float, default=0.0)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--lmbd', type=float, default=0.0)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--init_as_normal', type=int, default=0)
parser.add_argument('--reset_after_session', type=int, default=1)
parser.add_argument('--train_random_order', type=int, default=0)
parser.add_argument('--user_key', type=str, default='UserId')
parser.add_argument('--item_key', type=str, default='ItemId')
parser.add_argument('--session_key', type=str, default='SessionId')
parser.add_argument('--value_key', type=str, default='value')
parser.add_argument('--time_key', type=str, default='Time')
parser.add_argument('--save_to', type=str, default=None)
parser.add_argument('--load_from', type=str, default=None)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--rnd_seed', type=int, default=42)
# sampling
parser.add_argument('--n_sample', type=int, default=32)
parser.add_argument('--sample_alpha', type=float, default=0.0)
# evaluation
parser.add_argument('--eval_cutoff', type=int, default=20)
parser.add_argument('--eval_top_pop', type=int, default=0)
parser.add_argument('--eval_boot', type=int, default=-1)
parser.add_argument('--eval_file', type=str, default=None)
# embeddings
parser.add_argument('--item_embedding', type=int, default=50)
parser.add_argument('--load_item_embeddings', type=str, default=None)
# user bias parameters
parser.add_argument('--user_to_ses_act', type=str, default='tanh')
parser.add_argument('--user_propagation_mode', type=str, default='init')
parser.add_argument('--user_to_output', type=int, default=0)

args = parser.parse_args()


PATH_TO_TRAIN = ''
PATH_TO_TEST = ''

train_data = pd.read_csv(PATH_TO_TRAIN)
test_data = pd.read_csv(PATH_TO_TEST)


user_dic={}
userids = list(train_data.UserId.unique())
for i, j in enumerate(userids):
    user_dic[j] = i
train_data['UserId'] = train_data['UserId'].apply(lambda x: user_dic[x])
test_data['UserId'] = test_data['UserId'].apply(lambda x: user_dic[x])

item_dic = {}
itemids = list(train_data.ItemId.unique())
for i, j in enumerate(itemids):
    item_dic[j] = i
train_data['ItemId'] = train_data['ItemId'].apply(lambda x: item_dic[x])
test_data['ItemId'] = test_data['ItemId'].apply(lambda x: item_dic[x])

session_dic = {}
sessionids = list(train_data.SessionId.unique()) + list(test_data.SessionId.unique())
sessionids = list(set(sessionids))
for i, j in enumerate(sessionids):
    session_dic[j] = i
train_data['SessionId'] = train_data['SessionId'].apply(lambda x: session_dic[x])
test_data['SessionId'] = test_data['SessionId'].apply(lambda x: session_dic[x])


session_layers = [int(x) for x in args.session_layers.split(',')]
user_layers = [int(x) for x in args.user_layers.split(',')]

item_embedding_values = None
if args.load_item_embeddings is not None:
    item_embedding_values = np.load(args.load_item_embeddings)

model = RNN(session_layers=session_layers,
            user_layers=user_layers,
            loss=args.loss,
            item_embedding=args.item_embedding,
            init_item_embeddings=item_embedding_values,
            hidden_act=args.hidden_act,
            dropout_p_hidden_usr=args.dropout_p_hidden_usr,
            dropout_p_hidden_ses=args.dropout_p_hidden_ses,
            dropout_p_init=args.dropout_p_init,
            lmbd=args.lmbd,
            decay=args.decay,
            grad_cap=args.grad_cap,
            sigma=args.sigma,
            adapt=args.adapt,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            init_as_normal=bool(args.init_as_normal),
            reset_after_session=bool(args.reset_after_session),
            train_random_order=bool(args.train_random_order),
            n_epochs=args.n_epochs,
            user_key=args.user_key,
            session_key=args.session_key,
            item_key=args.item_key,
            time_key=args.time_key,
            seed=args.rnd_seed,
            user_to_session_act=args.user_to_ses_act,
            user_propagation_mode=args.user_propagation_mode,
            user_to_output=bool(args.user_to_output))

logger.info('session_layers: {}'.format(args.session_layers))
logger.info('user_layers: {}'.format(args.user_layers))
logger.info('loss: {}'.format(args.loss))
logger.info('hidden_act: {}'.format(args.hidden_act))
logger.info('batch_size: {}'.format(args.batch_size))
logger.info('dropout_p_hidden_usr: {}'.format(args.dropout_p_hidden_usr))
logger.info('dropout_p_hidden_ses: {}'.format(args.dropout_p_hidden_ses))
logger.info('dropout_p_init: {}'.format(args.dropout_p_init))
logger.info('init_as_normal: {}'.format(args.init_as_normal))
logger.info('lmbd: {}'.format(args.lmbd))
logger.info('grad_cap: {}'.format(args.grad_cap))
logger.info('sigma: {}'.format(args.sigma))
logger.info('decay (only for rmsprop): {}'.format(args.decay))
logger.info('rnd_seed: {}'.format(args.rnd_seed))
logger.info('')
logger.info('TRAINING:')
logger.info('adapt: {}'.format(model.adapt))
logger.info('learning_rate: {}'.format(args.learning_rate))
logger.info('momentum: {}'.format(args.momentum))
logger.info('n_epochs: {}'.format(args.n_epochs))
logger.info('train_random_order: {}'.format(args.train_random_order))
logger.info('reset_after_session: {}'.format(args.reset_after_session))
logger.info('n_epochs: {}'.format(args.n_epochs))
logger.info('early_stopping: {}'.format(args.early_stopping))
logger.info('save_to: {}'.format(args.save_to))
logger.info('load_from: {}'.format(args.load_from))
logger.info('')
logger.info('EMBEDDINGS:')
logger.info('item_embedding: {}'.format(args.item_embedding))
logger.info('load_item_embeddings: {}'.format(args.load_item_embeddings))
logger.info('')
logger.info('USER REPR. PROPAGATION:')
logger.info('user_to_session_act: {}'.format(args.user_to_ses_act))
logger.info('user_propagation_mode: {}'.format(args.user_propagation_mode))
logger.info('user_to_output: {}'.format(args.user_to_output))
logger.info('')
logger.info('EVALUATION:')
logger.info('eval_cutoff: {}'.format(args.eval_cutoff))
logger.info('eval_top_pop: {}'.format(args.eval_top_pop))
logger.info('eval_boot: {}'.format(args.eval_boot))
logger.info('eval_file: {}'.format(args.eval_file))

t0 = dt.now()
logger.info('Training started')
model.fit(train_data,
          valid_data=test_data if args.early_stopping else None,
          patience=3,
          margin=1.003,
          save_to=args.save_to,
          load_from=args.load_from)
logger.info('Training completed in {}'.format(dt.now() - t0))

logger.info('Evaluation started')
if args.eval_top_pop > 0:
    eval_items = train_data[args.item_key].value_counts()[:args.eval_top_pop].index
else:
    eval_items = None

evaluate_sessions_batch_hier_bootstrap(model,
                                       train_data,
                                       test_data,
                                       cut_off=5,
                                       output_rankings=True,
                                       bootstrap_length=args.eval_boot,
                                       batch_size=100,
                                       items=eval_items,
                                       session_key=args.session_key,
                                       user_key=args.user_key,
                                       item_key=args.item_key,
                                       time_key=args.time_key)
evaluate_sessions_batch_hier_bootstrap(model,
                                       train_data,
                                       test_data,
                                       cut_off=10,
                                       output_rankings=True,
                                       bootstrap_length=args.eval_boot,
                                       batch_size=100,
                                       items=eval_items,
                                       session_key=args.session_key,
                                       user_key=args.user_key,
                                       item_key=args.item_key,
                                       time_key=args.time_key)
evaluate_sessions_batch_hier_bootstrap(model,
                                       train_data,
                                       test_data,
                                       cut_off=20,
                                       output_rankings=True,
                                       bootstrap_length=args.eval_boot,
                                       batch_size=100,
                                       items=eval_items,
                                       session_key=args.session_key,
                                       user_key=args.user_key,
                                       item_key=args.item_key,
                                       time_key=args.time_key)

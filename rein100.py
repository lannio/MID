#import all

import os
import os.path as osp
import yaml
import argparse
import dill
import pdb
import time
import math
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils import data
from torch.autograd import Variable
from torch.nn import Module, Parameter, ModuleList, Linear

from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from easydict import EasyDict

import datetime, shutil, argparse, logging, sys
 
from dataset.preprocessing import get_node_timestep_data, collate 

# args temp
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_file',type=str, default='rein100.py')
    parser.add_argument('--config_file',type=str, default='./configs/baseline.yaml')

    parser.add_argument('--folder_date',type=str, default='0805')
    parser.add_argument('--dataset',type=str, default='eth')
    parser.add_argument('--exp',type=str, default='demo2')

    parser.add_argument('--read', default=False, type=bool)
    parser.add_argument('--model',type=str, default='/home/yaoliu/scratch/experiment/rein/test-0808-02/eth/baseline/95_model.pt')

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument("--eval_every", default=10, type=int)

    parser.add_argument("--num_steps", default=100, type=int)
    parser.add_argument("--num_ddim", default=10, type=int)
    parser.add_argument("--ddim_eta", default=0.0, type=float)
    parser.add_argument("--clip_denoised", default=False, type=bool)

    parser.add_argument("--loss_diffusion_rate", default=1., type=float)
    parser.add_argument("--loss_gau_rate", default=1000., type=float)
    parser.add_argument("--loss_mean_rate", default=1., type=float)
    parser.add_argument("--loss_left_gau", default=100., type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr2", default=0.001, type=float)

    parser.add_argument("--diffusion_sample_num", default=1, type=int)
    parser.add_argument("--point_dim", default=2, type=int)
    parser.add_argument("--pred_length", default=1, type=int)
    parser.add_argument("--end_list", default=20, type=int)
    parser.add_argument("--sample", default=20, type=int)

    parser.add_argument("--device", default=3, type=int)
    parser.add_argument("--isseed", default=True, type=bool)
    parser.add_argument('--seed', default=133, type=int)

    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--augment", default=True, type=bool)

    parser.add_argument('--data_dir',type=str, default='./processed_data_new')    
    parser.add_argument('--gpu_deterministic', default=False, type=bool, help='set cudnn in deterministic mode (slow)')


    return parser.parse_args()

# args 2
def get_traj_hypers():
    hypers = { 
    'state':
        {'PEDESTRIAN':
            {'position': ['x', 'y'],
             'velocity': ['x', 'y'],
             'acceleration': ['x', 'y']
            }
        },
    'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
    'edge_encoding': True,
    'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
    'edge_removal_filter': [1.0, 0.0],
    'dynamic_edges': 'yes',
    'incl_robot_node': False,
    'node_freq_mult_train': False,
    'node_freq_mult_eval': False,
    'scene_freq_mult_train': False,
    'scene_freq_mult_eval': False,
    'scene_freq_mult_viz': False,
    'use_map_encoding': False,
    }
    return hypers


# common function
def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format())

def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_output_dir(folder,dataset,exp):
    output_dir = os.path.join('/home/yaoliu/scratch/experiment/rein/' + folder, dataset, exp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


# data function
class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = 7 # 7
        self.max_ft = 12
        self.node_type_datasets = list() # 1-->1670
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        '''
        {'PEDESTRIAN': {'position': ['x', 'y'],
        'velocity': ['x', 'y'],
        'acceleration': ['x', 'y']}}
        '''
        self.pred_state = pred_state # {'PEDESTRIAN': {'velocity': ['x', 'y']}}
        self.hyperparams = hyperparams
        self.max_ht = 7 # 7
        self.max_ft = 12 #12

        self.augment = augment

        self.node_type = node_type # PEDESTRIAN
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index) # 1670
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type] # [(PEDESTRIAN, PEDESTRIAN)]

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs): # False False
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
    


class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        '''
            num_steps=100,
            beta_T=5e-2,
            mode='linear'
        '''
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps # 100

        self.beta_1 = beta_1 # 1e-4
        self.beta_T = beta_T # 5e-2
        self.mode = mode # 'linear'

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t).float()
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

def get_select_fde(selected_end,end):
    # fde=torch.mean(torch.norm((selected_end.view(256, -1, 2) - end.view(256,-1, 2)),dim=2,dim=[0])
    fde=torch.mean((torch.norm((selected_end.view(256, -1, 2) - end.view(256,-1, 2)),dim=2)),dim=0)
    # fde = F.mse_loss(end.contiguous().view(-1, 2), selected_end.contiguous().view(-1, 2), reduction='mean')
    return fde

def get_pred_loss(pred, selected_end, gt):
    # bs,12,5 bs,1,2, bs,12,2
    sx = torch.exp(pred[:, :, 2])  # sx
    sy = torch.exp(pred[:, :, 3])  # sy
    corr = torch.tanh(pred[:, :, 4])  # corr

    cov = torch.zeros(pred.shape[0], pred.shape[1], 2, 2).to('cuda')
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = pred[:, :, 0:2]
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    loss_gt = - mvn.log_prob(gt).sum()
    loss_mean = F.mse_loss(mean[:,-1,:].contiguous().view(-1, 2), selected_end.contiguous().view(-1, 2), reduction='mean')
    # loss=loss_gt/args.loss_gau_rate + loss_mean/args.loss_mean_rate

    return loss_gt, loss_mean

def get_pred_de(pred, gt):
    predlist=len(pred)
    kstep_V_pred_ls = []
    gt = gt.permute(1,0,2)*0.4
    pred = pred.permute(1,0,2)

    sx = torch.exp(pred[:, :, 2])  # sx
    sy = torch.exp(pred[:, :, 3])  # sy
    corr = torch.tanh(pred[:, :, 4])  # corr

    cov = torch.zeros(pred.shape[0], pred.shape[1], 2, 2).to('cuda')
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = pred[:, :, 0:2]
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

    KSTEPS=args.sample
    for i in range(KSTEPS-1):
        kstep_V_pred_ls.append(torch.cumsum((mvn.sample()*0.4), dim=0))  # cat [12, num_person, 2]
    kstep_V_pred_ls.append(torch.cumsum(mean*0.4, dim=0))
    kstep_V_pred_ls = torch.stack(kstep_V_pred_ls, dim=0) # [KSTEPS, 12, num_person, 2]

    # kstep_V_pred = np.concatenate([traj for traj in kstep_V_pred_ls], axis=1) # [12, KSTEPS * num_person, 2]

    """end of sampling"""

    V_y_rel_to_abs =  torch.cumsum((gt), dim=0) # [12, num_person, 2] speed???)

    ade=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs),dim=3),dim=0)[0],dim=[0,1])
    fde=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs)[:,-1,:,:],dim=2),dim=0)[0],dim=[0])
    return ade,fde



def find_end(gauss_param_tensor, coordinates_list):
    gauss_param_tensor = gauss_param_tensor[:,0,:]

    sx = torch.exp(gauss_param_tensor[:, 2])  # sx
    sy = torch.exp(gauss_param_tensor[:, 3])  # sy
    corr = torch.tanh(gauss_param_tensor[:, 4])  # corr
    cov = torch.zeros(gauss_param_tensor.shape[0], 2, 2).to('cuda')
    cov[:, 0, 0] = sx * sx
    cov[:, 0, 1] = corr * sx * sy
    cov[:, 1, 0] = corr * sx * sy
    cov[:, 1, 1] = sy * sy
    mean = gauss_param_tensor[:, 0:2] # bs,1,2
    # 创建MultivariateNormal分布对象
    gauss_distribution = MultivariateNormal(mean, cov)

    # 用来存储每个item的采样概率
    sampling_probs = []

    # 计算每个item与第一个tensor的采样概率
    for coordinates_tensor in coordinates_list:

        # 计算该item在第一个分布下的log概率之和
        log_prob_sum = gauss_distribution.log_prob(coordinates_tensor).sum()

        # 将采样概率存储到列表中
        sampling_probs.append(log_prob_sum)

    # 找到具有最大采样概率的item的索引
    max_prob_index = torch.argmax(torch.tensor(sampling_probs))

    # 选取最有可能是由第一个tensor采样得到的item
    selected_tensor = coordinates_list[max_prob_index]
    selected_sampling_probs= sampling_probs[max_prob_index]

    return selected_tensor, selected_sampling_probs


# main args
args = parse_args()
hyperparams = get_traj_hypers()


# main output-log  

output_dir = get_output_dir(args.folder_date, args.dataset, args.exp)
copy_source(args.run_file, output_dir)
copy_source(args.config_file, output_dir)


# set_gpu(args.device)
set_cuda(deterministic=args.gpu_deterministic)

if (args.isseed):
    set_seed(args.seed)

logger = setup_logging('job{}'.format(0), output_dir, console=True)
logger.info(args)


logger.info("--------build--------")


# data

logger.info("----dataset begin----")
train_data_path = osp.join(args.data_dir,args.dataset + "_train.pkl")
eval_data_path = osp.join(args.data_dir,args.dataset + "_test.pkl")
logger.info("train_data_path: "+ train_data_path)
logger.info("eval_data_path: "+ eval_data_path)

train_scenes = []
with open(train_data_path, 'rb') as f:
    train_env = dill.load(f, encoding='latin1')
train_scenes = train_env.scenes
train_dataset = EnvironmentDataset(train_env,
                                hyperparams['state'],
                                hyperparams['pred_state'],
                                scene_freq_mult=False,
                                node_freq_mult=False,
                                hyperparams=hyperparams,
                                min_history_timesteps=7,
                                min_future_timesteps=12,
                                return_robot=True)
train_data_loader = dict()
for node_type_data_set in train_dataset:
    node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                    collate_fn=collate,
                                                    pin_memory = True,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.shuffle,
                                                    num_workers=0,
                                                    drop_last=True)
    train_data_loader[node_type_data_set.node_type] = node_type_dataloader

eval_scenes = []
with open(eval_data_path, 'rb') as f:
    eval_env = dill.load(f, encoding='latin1')
eval_scenes = eval_env.scenes
eval_dataset = EnvironmentDataset(eval_env,
                                hyperparams['state'],
                                hyperparams['pred_state'],
                                scene_freq_mult=False,
                                node_freq_mult=False,
                                hyperparams=hyperparams,
                                min_history_timesteps=7,
                                min_future_timesteps=12,
                                return_robot=True)
eval_data_loader = dict()
for node_type_data_set in eval_dataset:
    node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                    collate_fn=collate,
                                                    pin_memory=True,
                                                    batch_size=args.eval_batch_size,
                                                    shuffle=args.shuffle,
                                                    num_workers=0,
                                                    drop_last=True)
    eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

logger.info("----dataset end----")


logger.info("----model begin----")


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias

        return ret
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x
    
class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)
        # score_vec is of shape (batch, num_enc_states)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs
    
class Model_Dim_Up(Module):
    def __init__(self):
        super(Model_Dim_Up, self).__init__()
        # bs, 1, 2 --> bs, 1, 5

        # self.cnn_up = MLP(input_dim = 2, output_dim = 5, hidden_size=[16,64])
        self.cnn_up = nn.Conv1d(2, 5, 1, padding=0)


    def forward(self, x):
        x=x.permute(0,2,1)
        x = self.cnn_up(x) 
        x = x.permute(0,2,1)

        return x
    
# model1=Model_Dim_Up()
# data1=torch.randn(64,1,2)
# out=model1(data1)
# print(out.size())
# # torch.Size([64, 1, 5])




class Model_Encoder_His(Module):
    def __init__(self):
        super(Model_Encoder_His, self).__init__()
        # bs, time, 6 --> bs, 128

        self.encoder_his =nn.LSTM(input_size=6, hidden_size=128, batch_first=True).cuda()
        self.dropout = nn.Dropout(p=0.25)


    def forward(self, node_history_st):
    
        his_feat, _ = self.encoder_his(node_history_st)  
        his_feat = self.dropout(his_feat)
        his_feat = his_feat[:,-1,:]

        return his_feat


class Model_Encoder_Nei(Module):
    def __init__(self):
        super(Model_Encoder_Nei, self).__init__()
        # bs, time, 6 --> bs, 128

        self.encoder_nei =nn.LSTM(input_size=12, hidden_size=128, batch_first=True).cuda()
        self.encoder_combine=AdditiveAttention(encoder_hidden_state_dim=128, decoder_hidden_state_dim=128).cuda()
        self.dropout = nn.Dropout(p=0.25)

        self.state = hyperparams['state']

    def forward(self, node_history_st, neighbors, neighbors_edge_value, edge_type, his_feat):
    

        edge_states_list = list()  
        for i, neighbor_states in enumerate(neighbors):  
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                ) # 6
                edge_states_list.append(torch.zeros(1, 8, neighbor_state_length).cuda())
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).cuda())
        
        op_applied_edge_states_list = list()
        for neighbors_state in edge_states_list:
            op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0)) #  list of [max_ht, state_dim] torch.Size([8, 6])
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0) # torch.Size([256, 8, 6])

        op_applied_edge_mask_list = list()
        for edge_value in neighbors_edge_value:
            op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.cuda(), dim=0, keepdim=True), max=1.))
        combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0) # torch.Size([256, 1])

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        nei_feat, _ = self.encoder_nei(joint_history) 
        nei_feat = self.dropout(nei_feat)
        nei_feat = nei_feat[:,-1,:]

        nei_feat = nei_feat * combined_edge_masks

        nei_feats = torch.stack([nei_feat], dim=1)

        combined_feat, _ = self.encoder_combine(nei_feats, his_feat)
        combined_feat = self.dropout(combined_feat)


        return combined_feat

# model1=Model_Encoder_His().cuda()
# model2=Model_Encoder_Nei().cuda()

# for node_type, data_loader in train_data_loader.items():
#     break
# for batch in data_loader:
#     break
# edge_type=train_env.get_edge_types()[0]
# (first_history_index,
#     x_t, y_t, x_st_t, y_st_t, # y_t torch.Size([256, 12, 2])
#     neighbors_data_st,
#     neighbors_edge_value,
#     robot_traj_st_t,
#     map) = batch

# his_f=model1(x_st_t.cuda())
# print(his_f.size())
# # torch.Size([256, 128])
# nei_f=model2(x_st_t.cuda(),restore(neighbors_data_st)[edge_type],restore(neighbors_edge_value)[edge_type],edge_type, his_f)
# print(nei_f.size())
# # torch.Size([256, 128])



class Model_backbone(Module):
    def __init__(self):
        super(Model_backbone, self).__init__()

        context_dim=256
        dim=5
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(dim,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, dim, context_dim+3)
        #self.linear = nn.Linear(128,2)


    def forward(self, endpoint_feat, beta, guide):
        # bs,1,5 bs,8,256

        batch_size = endpoint_feat.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        guide = guide.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, guide], dim=-1)    # (B, 1, F+3)
        endpoint_feat = self.concat1(ctx_emb,endpoint_feat)
        final_emb = endpoint_feat.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

# model1=Model_backbone()
# endpoint_feat=torch.randn(64,1,5)
# beta=torch.randn(64)
# guide=torch.randn(64,256)
# model1(endpoint_feat,beta,guide).size()
# # torch.Size([64, 1, 5])

class Model_his_to_end(Module):
    def __init__(self):
        super(Model_his_to_end, self).__init__()
        # bs, 128 --> bs, 5

        self.encoder_his = MLP(input_dim = 128, output_dim = 5, hidden_size=[32, 8])

    def forward(self, his_feat):
        end_list=[]
        end_feat = self.encoder_his(his_feat) # bs,5

        sx = torch.exp(end_feat[:, 2])  # sx
        sy = torch.exp(end_feat[ :, 3])  # sy
        corr = torch.tanh(end_feat[ :, 4])  # corr

        cov = torch.zeros(end_feat.shape[0], 2, 2).to('cuda')
        cov[:, 0, 0] = sx * sx
        cov[:, 0, 1] = corr * sx * sy
        cov[:, 1, 0] = corr * sx * sy
        cov[:, 1, 1] = sy * sy
        mean = end_feat[:, 0:2]
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

        for i in range(args.end_list):

            end_list.append(mvn.sample())

        return end_list
    #  10* torch.Size([64,2])
# model1=Model_his_to_end()
# data1=torch.randn(64,128)
# model1(data1)[0].size()


class Model_all_to_pred(Module):
    def __init__(self):
        super(Model_all_to_pred, self).__init__()

        context_dim=256

        dim=2

        self.his_pred = MLP(input_dim = 8, output_dim = 11, hidden_size=[32,128])

        self.encoder_end = MLP(input_dim = 2, output_dim = 128, hidden_size=[8,32])

        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(dim,2*context_dim,context_dim)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim)
        self.linear = ConcatSquashLinear(context_dim//2, 5, context_dim)

    def forward(self, his, end, his_feat, nei_feat):
        bs=end.size()[0]

        his=his[:,:,2:4]
        his_p=his.permute(0,2,1)
        his_pred=self.his_pred(his_p)
        his_pred=his_pred.permute(0,2,1)
        his_all=torch.concat((his_pred,end.view(bs, 1, -1)),dim=1) #bs,12,2
        ctx_emb=torch.concat((his_feat,nei_feat),dim=1).view(bs, 1, -1)

        pred_feat = self.concat1(ctx_emb,his_all)
        final_emb = pred_feat.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

# model1=Model_all_to_pred()
# his=torch.randn(64,8,2)
# end=torch.randn(64,2)
# his_feat=torch.randn(64,128)
# nei_feat=torch.randn(64,128)
# model1(his,end,his_feat,nei_feat).size()
# # torch.Size([64, 12, 5])


class Model_diffusion(Module):
    def __init__(self):
        super(Model_diffusion, self).__init__()

        self.backbone=Model_backbone().cuda()    
        self.var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_T=5e-2,
                mode='linear',
            ).cuda()    


    def get_loss(self, model_up, endpoint, his,nei,mask,edge_type,his_en,nei_en):

        endpoint_oir=torch.clone(endpoint) #bs,1,2
        endpoint=model_up(endpoint)#bs,1,5


        sx = torch.exp(endpoint[:, :, 2])  # sx
        sy = torch.exp(endpoint[:, :, 3])  # sy
        corr = torch.tanh(endpoint[:, :, 4])  # corr
        cov = torch.zeros(endpoint.shape[0], endpoint.shape[1], 2, 2).to('cuda')
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = endpoint[:, :, 0:2] # bs,1,2
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        loss_gau = - mvn.log_prob(endpoint_oir).sum()
        loss_mean = F.mse_loss(mean.contiguous().view(-1, 2), endpoint_oir.contiguous().view(-1, 2), reduction='mean')

        his_feat=his_en(his)
        nei_feat=nei_en(his, nei, mask, edge_type, his_feat)
        guide= torch.concat((his_feat,nei_feat),dim=1)

        batch_size, _, point_dim = endpoint.size() #$ bs,1,5

        t = self.var_sched.uniform_sample_t(batch_size) # 256 t 

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(endpoint).cuda()  # (B, N, d) torch.Size([256, 12, 2])

        e_theta = self.backbone(c0 * endpoint + c1 * e_rand, beta, guide) # torch.Size([256, 12, 2])

        loss_diffusion = F.mse_loss(e_theta.contiguous().view(-1, point_dim), e_rand.contiguous().view(-1, point_dim), reduction='mean')
        
        return loss_diffusion ,loss_gau ,loss_mean, endpoint, his_feat,nei_feat,guide



    def sample(self, model_up, his,nei,mask,edge_type,his_en,nei_en):
        gau_up=model_up

        traj_list = []
        point_dim=args.point_dim
        num_points=args.pred_length
        self.alphas_cumprod = self.var_sched.alpha_bars

        his_feat=his_en(his)
        nei_feat=nei_en(his, nei, mask, edge_type, his_feat)
        guide= torch.concat((his_feat,nei_feat),dim=1)

        for diff_sample_num in range(args.diffusion_sample_num):
        
            batch_size = guide.size(0)

            ddim_timesteps=args.num_ddim
            ddim_eta=args.ddim_eta
            clip_denoised=args.clip_denoised

            c = self.var_sched.num_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.var_sched.num_steps, c)))
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            ddim_timestep_seq = ddim_timestep_seq + 1
            # previous sequence
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])



            sample_img = torch.randn([batch_size, num_points, point_dim]).to(guide.device)
            sample_img=gau_up(sample_img)


            ddim_timesteps_test = ddim_timesteps
            # ddim_timesteps_test = self.config.ddim_timesteps_test
            for i in reversed(range(0, ddim_timesteps_test)) :
                t = torch.full((batch_size,), ddim_timestep_seq[i], device=guide.device, dtype=torch.long)
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=guide.device, dtype=torch.long)
                
                # 1. get current and previous alpha_cumprod
                
                alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
                alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
        
                # 2. predict noise using model
                beta = self.var_sched.betas[[t[0].item()]*batch_size]
                pred_noise = self.backbone(sample_img, beta, guide)
                
                # 3. get the predicted x_0
                pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                if clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                
                # 4. compute variance: "sigma_t(η)" -> see formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
                
                # 5. compute "direction pointing to x_t" of formula (12)
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
                
                # 6. compute x_{t-1} of formula (12)
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

                sample_img = x_prev.detach()
            traj_list.append(sample_img)

        return traj_list,his_feat,nei_feat,guide
    


class OUR():
    def __init__(self):
        super(OUR, self).__init__()
        self.gau_up=Model_Dim_Up().cuda() 
        self.his_en=Model_Encoder_His().cuda() 
        self.nei_en=Model_Encoder_Nei().cuda() 
        self.model_diffuion=Model_diffusion().cuda() 

        self.model_end=Model_his_to_end().cuda()
        self.model_pred=Model_all_to_pred().cuda()

        self.train_dataset=train_dataset
        self.hyperparams = hyperparams
        

        self.optimizer_right = optim.Adam([{'params': self.gau_up.parameters()},
                                     {'params': self.model_diffuion.parameters()},
                                     {'params': self.his_en.parameters()},
                                     {'params': self.nei_en.parameters()},
                                    ],
                                    lr=args.lr)
        self.scheduler_right = optim.lr_scheduler.ExponentialLR(self.optimizer_right,gamma=args.gamma)

        self.optimizer_left = optim.Adam([{'params': self.model_end.parameters()},
                                     {'params': self.model_pred.parameters()}
                                    ],
                                    lr=args.lr2)
        self.scheduler_left = optim.lr_scheduler.ExponentialLR(self.optimizer_left,gamma=args.gamma)
        
    def train(self):
        self.gau_up.train()
        self.his_en.train()
        self.nei_en.train()
        self.model_diffuion.train()

        self.model_end.train()
        self.model_pred.train()

        ade_final=9999
        fde_final=9999
        ade_avg = 999
        fde_avg = 999
        ade_epoch=0
        fde_epoch=0
        ftimesum=0.
        btimesum=0.
        sample_count=0

        for epoch in range(1, args.epochs + 1):
            self.gau_up.train()
            self.his_en.train()
            self.nei_en.train()
            self.model_diffuion.train()

            self.model_end.train()
            self.model_pred.train()

            start_time_f = time.time()
            self.train_dataset.augment = args.augment


            for node_type, data_loader in train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                right_loss=0.0
                right_loss_diff=0.0
                right_loss_gau=0.0
                right_loss_mean=0.0

                left_loss=0.0
                left_loss_gau=0.0
                left_loss_mean=0.0
                select_fde_sum = 0.0

                count=0
                for batch in pbar:
                    edge_type=train_env.get_edge_types()[0]
                    (first_history_index,
                        x_t, y_t, x_st_t, y_st_t, # y_t torch.Size([256, 12, 2])
                        neighbors_data_st,
                        neighbors_edge_value,
                        robot_traj_st_t,
                        map) = batch

                    self.his=(x_st_t).cuda()
                    self.gt=(y_st_t).cuda()
                    self.end=self.gt[:,11:12,:]
                    self.nei=restore(neighbors_data_st)[edge_type]
                    self.nei_mask=restore(neighbors_edge_value)[edge_type]

                    self.optimizer_left.zero_grad()

                    right_loss1,right_loss2,right_loss3, end_feat, his_feat,nei_feat,guide = self.model_diffuion.get_loss(self.gau_up, self.end, self.his,self.nei, self.nei_mask,edge_type,self.his_en,self.nei_en)
                    train_loss_right = right_loss1*args.loss_diffusion_rate+right_loss2/args.loss_gau_rate+right_loss3*args.loss_mean_rate

                    right_loss = right_loss + train_loss_right.item()
                    right_loss_diff = right_loss_diff + right_loss1.item()
                    right_loss_gau = right_loss_gau + right_loss2.item()
                    right_loss_mean = right_loss_mean + right_loss3.item()


                    end_list=self.model_end(his_feat)
                    selected_end, probs=find_end(end_feat, end_list)

                    select_fde=get_select_fde(selected_end,self.end)
                    select_fde_sum=select_fde_sum+select_fde.item()
                    
                    pred=self.model_pred(self.his, selected_end, his_feat, nei_feat)
                    left_loss1, left_loss2 = get_pred_loss(pred, selected_end, self.gt)

                    train_loss_left=left_loss1/args.loss_left_gau+train_loss_right
                    pbar.set_description(f"Epoch {epoch}, {node_type} Left-MSE: {train_loss_left.item():.2f}")
                    left_loss = left_loss + train_loss_left.item()
                    left_loss_gau = left_loss_gau + left_loss1.item()
                    left_loss_mean = left_loss_mean + left_loss2.item()


                    count = count+1

                    train_loss_left.backward()
                    self.optimizer_left.step()



            if args.dataset == "eth":
                select_fde_sum=select_fde_sum/0.6
            elif args.dataset == "sdd":
                select_fde_sum=select_fde_sum* 50


            end_time_f = time.time()
            ftime= end_time_f - start_time_f
            ftimesum = ftimesum+ftime
            logger.info(f"Epoch {epoch}, {node_type} Right-MSE: {(right_loss/count):.2f}, loss1 MSE: {(right_loss_diff/count):.2f}, loss2 MSE: {(right_loss_gau/count):.2f}, loss3 MSE: {(right_loss_mean/count):.2f}, train_time: {(ftime):.2f}, train_time_avg: {(ftimesum/epoch):.2f}")
            logger.info(f"Epoch {epoch}, {node_type} Left-MSE: {(left_loss/count):.2f}, loss1 gau: {(left_loss_gau/count):.2f}, loss2 MSE: {(left_loss_mean/count):.2f}, train_time: {(ftime):.2f}, train_time_avg: {(ftimesum/epoch):.2f}")
            logger.info(f"DE {epoch}, {node_type},select_FDE: {(select_fde_sum/count):.2f} ")

            
            # if ((epoch % args.eval_every == 0) and (epoch > 0)) or epoch==1:
            with torch.no_grad():
                self.train_dataset.augment = False
                start_time_b = time.time()
                self.gau_up.eval()
                self.his_en.eval()
                self.nei_en.eval()
                self.model_diffuion.eval()

                self.model_end.eval()
                self.model_pred.eval()

                node_type = "PEDESTRIAN"

                ade_sum=0.0
                fde_sum=0.0
                test_count=0

                for node_type_test, data_loader_test in eval_data_loader.items():
                    pbar2 = tqdm(data_loader_test, ncols=80)
                    for test_batch in pbar2:
                        (first_history_index,
                            x_t, y_t, x_st_t, y_st_t, # y_t torch.Size([256, 12, 2])
                            neighbors_data_st,
                            neighbors_edge_value,
                            robot_traj_st_t,
                            map) = test_batch

                        self.test_his=(x_st_t).cuda()
                        self.test_gt=(y_st_t).cuda()
                        self.test_end=self.test_gt[:,11:12,:]
                        self.test_nei=restore(neighbors_data_st)[edge_type]
                        self.test_nei_mask=restore(neighbors_edge_value)[edge_type]

                        traj_pred_list,his_feat,nei_feat,guide = self.model_diffuion.sample(self.gau_up, self.test_his,self.test_nei,self.test_nei_mask,edge_type,self.his_en,self.nei_en) # bs,1,5
                        traj_pred=traj_pred_list[0]
                        
                        end_list=self.model_end(his_feat)
                        selected_end, probs=find_end(traj_pred, end_list)
                        pred=self.model_pred(self.test_his, selected_end, his_feat, nei_feat)
                        ade, fde = get_pred_de(pred, self.test_gt)
                        
                        ade_sum=ade_sum+ade
                        fde_sum=fde_sum+fde
                        test_count=test_count+1

                ade_avg=ade_sum/test_count
                fde_avg=fde_sum/test_count



                if args.dataset == "eth":
                    ade_avg = ade_avg/0.6
                    fde_avg = fde_avg/0.6
                elif args.dataset == "sdd":
                    ade_avg = ade_avg * 50
                    fde_avg = fde_avg * 50

                end_time_b = time.time()
                btime= end_time_b - start_time_b
                btimesum = btimesum+btime
                sample_count=sample_count+1

                
                logger.info(f"{args.folder_date} {args.dataset}  {args.exp}  :Best of 20: Epoch {epoch} (Train) ADE: {ade_avg} FDE: {fde_avg}, sample_time: {(btime):.2f}, sample_time_avg: {(btimesum/sample_count):.2f}")


                save_path = output_dir +'/'+ str(epoch)+ '_model.pt'
                torch.save({
                            'gau_up': self.gau_up.state_dict(),
                            'his_en': self.his_en.state_dict(),
                            'nei_en': self.nei_en.state_dict(),
                            'model_diffuion': self.model_diffuion.state_dict(),       
                            'model_end': self.model_end.state_dict(),  
                            'model_pred': self.model_pred.state_dict(),
                            'optimizer_right': self.optimizer_right.state_dict(),
                            'optimizer_left': self.optimizer_left.state_dict(),
                            }, save_path)
                logger.info("Saved model to:\n{}".format(save_path))
        
            


            if (ade_final>ade_avg):
                ade_final = ade_avg
                ade_epoch = epoch
            if (fde_final>fde_avg):
                fde_final=fde_avg
                fde_epoch = epoch
            print(f"######## Best Of 20 (Train): ADE: {ade_epoch} -- {ade_final} FDE: {fde_epoch} -- {fde_final}")
            logger.info(f"######## Best Of 20 (Train): ADE: {ade_epoch} -- {ade_final} FDE: {fde_epoch} -- {fde_final}")


def main():
    agent = OUR()
    agent.train()


if __name__ == '__main__':
    main()

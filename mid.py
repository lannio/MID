import os
import argparse
import torch
import logging
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation
import time

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


   
class MID():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        ade_final=9999
        fde_final=9999
        ade = 999
        fde = 999
        ade_epoch=0
        fde_epoch=0
        ftimesum=0.
        btimesum=0.
        sample_count=0
        for epoch in range(1, self.config.epochs + 1):
            start_time_f = time.time()
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                batch_loss=0.0
                batch_loss1=0.0
                batch_loss2=0.0
                batch_loss3=0.0
                count=0
                for batch in pbar:

                    self.optimizer.zero_grad()
                    # logging.info(f"Begin_Train_model:{epoch}")
                    [train_loss,loss1,loss2,loss3] = self.model.get_loss(batch, node_type, logging)
                    # logging.info(f"End_Train_model:{epoch}")
                    pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                    count = count+1
                    batch_loss = batch_loss + train_loss.item()
                    batch_loss1 = batch_loss1 + loss1.item()
                    batch_loss2 = batch_loss2 + loss2.item()
                    batch_loss3 = batch_loss3 + loss3.item()
                    train_loss.backward()
                    self.optimizer.step()
            end_time_f = time.time()
            ftime= end_time_f - start_time_f
            ftimesum = ftimesum+ftime
            # print(f"Epoch {epoch}, {node_type} MSE: {(batch_loss/count):.2f}, loss1 MSE: {(batch_loss1/count):.2f}, loss2 MSE: {(batch_loss2/count):.2f}, loss3 MSE: {(batch_loss3/count):.2f}, train_time: {(ftime):.2f}, train_time_avg: {(ftimesum/epoch):.2f}")
            logging.info(f"Epoch {epoch}, {node_type} MSE: {(batch_loss/count):.2f}, loss1 MSE: {(batch_loss1/count):.2f}, loss2 MSE: {(batch_loss2/count):.2f}, loss3 MSE: {(batch_loss3/count):.2f}, train_time: {(ftime):.2f}, train_time_avg: {(ftimesum/epoch):.2f}")

            self.train_dataset.augment = False
            if ((epoch % self.config.eval_every == 0) and (epoch > 0)) or epoch==1:
                start_time_b = time.time()
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.hyperparams['prediction_horizon'] #12
                max_hl = self.hyperparams['maximum_history_length'] #7

                ade_sum=0.0
                fde_sum=0.0
                count=0

 
                for node_type_test, data_loader_test in self.eval_data_loader.items():
                    pbar2 = tqdm(data_loader_test, ncols=80)
                    for test_batch in pbar2:
                        traj_pred = self.model.generate(test_batch, node_type_test, logging, num_points=12, sample=self.config.sample, sample_out=self.config.sample_out,bestof=True) # B * 20 * 12 * 2

                        if (self.config.newLoss):
                            if (self.config.newLoss1010):
                                batch_error_dict = evaluation.compute_new_de1010(traj_pred,test_batch[2].to('cuda'),self.config)
                            else:
                                batch_error_dict = evaluation.compute_new_de(traj_pred,test_batch[2].to('cuda'),self.config)
                        else:
                            batch_error_dict = evaluation.compute_nogauss_de(traj_pred,test_batch[2].to('cuda'),self.config)
                        ade_sum=ade_sum+batch_error_dict[0]
                        fde_sum=fde_sum+batch_error_dict[1]
                        count=count+1
                            # lanni
                            # eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                            # eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))

                    ade=ade_sum/count
                    fde=fde_sum/count

                # else:
                #     for i, scene in enumerate(self.eval_scenes):
                #         # logging.info(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                #         print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                #         for t in tqdm(range(0, scene.timesteps, 10)):
                #             timesteps = np.arange(t,t+10)
                #             batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                #                         pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                #                         min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                #                         max_ft=12, hyperparams=self.hyperparams)
                #             if batch is None:
                #                 continue
                #             test_batch = batch[0]
                #             nodes = batch[1]
                #             timesteps_o = batch[2]
                #             traj_pred = self.model.generate(test_batch, node_type, logging, num_points=12, sample=self.config.sample, sample_out=self.config.sample_out, bestof=True) # B * 20 * 12 * 2
                #             # torch.Size([20, 16, 12, 2])
                #             predictions = traj_pred
                #             predictions_dict = {}
                #             for i, ts in enumerate(timesteps_o):
                #                 if ts not in predictions_dict.keys():
                #                     predictions_dict[ts] = dict()
                #                 predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3)) # 20,bs,12,2

                #             batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                #                                                                 scene.dt,
                #                                                                 max_hl=max_hl,
                #                                                                 ph=ph,
                #                                                                 node_type_enum=self.eval_env.NodeType,
                #                                                                 kde=False,
                #                                                                 map=None,
                #                                                                 best_of=True,
                #                                                                 prune_ph_to_future=True)

                #             eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                #             eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



                #     ade = np.mean(eval_ade_batch_errors)
                #     fde = np.mean(eval_fde_batch_errors)

                if self.config.dataset == "eth":
                    ade = ade/0.6
                    fde = fde/0.6
                elif self.config.dataset == "sdd":
                    ade = ade * 50
                    fde = fde * 50

                end_time_b = time.time()
                btime= end_time_b - start_time_b
                btimesum = btimesum+btime
                sample_count=sample_count+1

                # print(f"Epoch {epoch} Best Of 20 (Train): ADE: {ade} FDE: {fde}, sample_time: {(btime):.2f}, sample_time_avg: {(btimesum/sample_count):.2f}")
                logging.info(f"{self.config.folder} {self.config.dataset}  {self.config.exp}  :Best of 20: Epoch {epoch} (Train) ADE: {ade} FDE: {fde}, sample_time: {(btime):.2f}, sample_time_avg: {(btimesum/sample_count):.2f}")

                # Saving model
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                 }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

                self.model.train()
        
            if (ade_final>ade):
                ade_final = ade
                ade_epoch = epoch
            if (fde_final>fde):
                fde_final=fde
                fde_epoch = epoch
        print(f"######## Best Of 20 (Train): ADE: {ade_epoch} -- {ade_final} FDE: {fde_epoch} -- {fde_final}")
        logging.info(f"######## Best Of 20 (Train): ADE: {ade_epoch} -- {ade_final} FDE: {fde_epoch} -- {fde_final}")



    def eval(self):
        epoch = self.config.eval_at



        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']


        for i, scene in enumerate(self.eval_scenes):
            logging.info(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                               min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                               max_ft=12, hyperparams=self.hyperparams)
                if batch is None:
                    continue
                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]
                # logging.info(f"Begin_EVAL_model:{epoch}")
                traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=self.config.sample, sample_out=self.config.sample_out,bestof=True) # B * 20 * 12 * 2
                # logging.info(f"Begin_EVAL_model:{epoch}")

                predictions = traj_pred
                predictions_dict = {}
                for i, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))



                batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=self.eval_env.NodeType,
                                                                       kde=False,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50


        print(f"Epoch {epoch} Best Of 20 (--------EVAL--------): ADE: {ade} FDE: {fde}")
        logging.info(f"Best of 20: Epoch {epoch} (--------EVAL--------) ADE: {ade} FDE: {fde}")
        #self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")


    def _build(self):
        self._build_dir()

        # logging.info("Begin_encoder_config")
        self._build_encoder_config() # train_env eval_env
        # logging.info("Begin_encoder")
        self._build_encoder()
        # logging.info("Begin_model")
        self._build_model()
        # logging.info("Begin_train_loader")
        self._build_train_loader()
        # logging.info("Begin_eval_loader")
        self._build_eval_loader()

        # logging.info("Begin_optimizer")
        self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")
        logging.info("End_bulid")
        logging.info("\n")

        logging.info("Dataset on:")
        logging.info(self.config.dataset)
        logging.info("\n")

    def _build_dir(self):
        self.model_dir = osp.join("/home/yaoliu/scratch/experiment/diffusion/",self.config.folder,self.config.dataset,self.config.exp)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        set_logger(log_dir)
        # self.log = logging.getLogger()
        # self.log.setLevel(logging.INFO)
        # handler = logging.FileHandler(log_dir)
        # handler.setLevel(logging.INFO)
        # self.log.addHandler(handler)

        logging.info("Config:")
        logging.info(self.config)
        logging.info("\n")


        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=self.config.gamma)
        print("> Optimizer built!")

    def _build_encoder_config(self):

        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registrain_dataset.augment = config.augmenttar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = self.config.model_name
            self.checkpoint = torch.load(checkpoint_dir, map_location = "cpu")
            # print(self.config.)
            print("> eval_mode!!!!!!!!!!!")

            self.registrar.load_models(self.checkpoint['encoder'])
            # epoch = self.config.eval_at
            # checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            # self.checkpoint = torch.load(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"), map_location = "cpu")

            # self.registrar.load_models(self.checkpoint['encoder'])

        # 528528
        checkpoint_dir = self.config.model_name
        self.checkpoint = torch.load(checkpoint_dir, map_location = "cpu")
        self.registrar.load_models(self.checkpoint['encoder'])

        # ? lanni train_env 
        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')
        # self.train_env=self.eval_env

    def _build_encoder(self):
        # ? lanni scene Trajectron
        self.encoder = Trajectron(self.config, self.registrar, self.hyperparams, "cuda")

        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)

        self.model = model.cuda()
        if self.config.eval_mode:
            print("> ~~~~~~~~~~~~~~~~~~~~~~~!")
            self.model.load_state_dict(self.checkpoint['ddpm'])
        #528528
        self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        '''
            'state':
                {'PEDESTRIAN':
                    {'position': ['x', 'y'],
                    'velocity': ['x', 'y'],
                    'acceleration': ['x', 'y']
                    }
                },
            'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
            'log_histograms': False,
            'dynamic_edges': 'yes',
            'edge_state_combine_method': 'sum',
            'edge_influence_combine_method': 'attention',
            'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
            'edge_removal_filter': [1.0, 0.0],
            'offline_scene_graph': 'yes',
            'incl_robot_node': False,
            'node_freq_mult_train': False,
            'node_freq_mult_eval': False,
            'scene_freq_mult_train': False,
            'scene_freq_mult_eval': False,
            'scene_freq_mult_viz': False,
            'edge_encoding': True,
        '''
        self.train_dataset = EnvironmentDataset(train_env,
                                           self.hyperparams['state'],
                                           self.hyperparams['pred_state'],
                                           scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                           node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                           hyperparams=self.hyperparams,
                                           min_history_timesteps=7,
                                           min_future_timesteps=self.hyperparams['prediction_horizon'],
                                           return_robot=not self.config.incl_robot_node)
        # b=[]
        # for ii in range(len(a.node_type_datasets[0])): 
        #     x=torch.mean(a.node_type_datasets[0][ii][2],dim=0)[0]
        #     y=torch.mean(a.node_type_datasets[0][ii][2],dim=0)[1]
        #     if(torch.abs(x)>0.2 or torch.abs(y)>0.2):
        #         b.append( a.node_type_datasets[0][ii])

        # for i in range(self.train_dataset.node_type_datasets[0].len):
        #     self.train_dataset.node_type_datasets[0][i][1]=torch.nan_to_num(self.train_dataset.node_type_datasets[0][i][1])
        #     self.train_dataset.node_type_datasets[0][i][3]=torch.nan_to_num(self.train_dataset.node_type_datasets[0][i][3])

        self.train_data_loader = dict()
        '''
            return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple)
        '''
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory = True,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=self.config.shuffle,
                                                         num_workers=self.config.preprocess_workers,
                                                         drop_last=True)
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader


    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=7,
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            # for i in range(self.eval_dataset.node_type_datasets[0].len):
            #     self.eval_dataset.node_type_datasets[0][i][1]=torch.nan_to_num(self.eval_dataset.node_type_datasets[0][i][1])
            #     self.eval_dataset.node_type_datasets[0][i][3]=torch.nan_to_num(self.eval_dataset.node_type_datasets[0][i][3])
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=self.config.shuffle,
                                                             num_workers=config.preprocess_workers,
                                                             drop_last=True)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")

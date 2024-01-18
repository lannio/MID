import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb

'''
diffnet: TransformerConcatLinear
encoder_dim: 256
tf_layer: 3
epochs: 90
batch_size: 256
eval_batch_size: 256
k_eval: 25
seed: 123
eval_every: 1
# Testing
eval_at: 70
eval_mode: False
'''
class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet) #TransformerConcatLinear
        inpuit_dim=2
        if (self.config.newLoss):
            inpuit_dim=5
        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=inpuit_dim, config=self.config, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False), # 2, 256, 3, False
            var_sched = VarianceSchedule(
                num_steps=self.config.num_steps,
                beta_T=5e-2,
                mode='linear',
                config = self.config

            )
        )

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z

    def generate(self, batch, node_type, logging, num_points, sample, sample_out, bestof, flexibility=0.0, ret_traj=False):
        # self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True)
        # logging.info(f"Begin_feat_x_encoded")
        
        encoded_x = self.encoder.get_latent(batch, node_type) # torch.Size([16, 256])

        # if (self.config.newLoss):
        if (self.config.ddim):
            predicted_y_vel =  self.diffusion.sample_ddim_new(num_points, encoded_x,sample,sample_out, bestof, flexibility=flexibility, ret_traj=ret_traj) 
        else:
            predicted_y_vel =  self.diffusion.sample_ddpm_new(num_points, encoded_x,sample,sample_out, bestof, flexibility=flexibility, ret_traj=ret_traj) 
        return predicted_y_vel
        # else:
        #     dynamics = self.encoder.node_models_dict[node_type].dynamic
        #     if (self.config.ddim):
        #         predicted_y_vel =  self.diffusion.sample_ddim(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj) # torch.Size([20, 16, 12, 2])
        #     else:
        #         predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj) # torch.Size([20, 16, 12, 2])
        #     predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        #     return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type, logging):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t, # y_t torch.Size([256, 12, 2])
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        # logging.info(f"Begin_feat_x_encoded")
        feat_x_encoded = self.encode(batch,node_type) # B * 64 // torch.Size([256, 256])
        # logging.info(f"End_feat_x_encoded")
        # logging.info(f"Begin_get_loss")
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        # logging.info(f"End_get_loss")
        return loss

# 1 原始 
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb

class VarianceSchedule(Module):

    def __init__(self, num_steps, config, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        '''
            num_steps=100,
            beta_T=5e-2,
            mode='linear'
        '''
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps # 100
        self.num_ddim = config.num_ddim # 50
        self.config = config
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

class TrajNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(2, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 2, context_dim+3),

        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class CNNConcatLinear(Module): # this

    def __init__(self, point_dim, config, context_dim, tf_layer, residual):
        # 2, 256, 3, False
        super().__init__()
        self.config = config
        # context_dim = self.config.encoder_dim/self.config.encoder_dim_cnn
        # self.residual = residual
        self.pos_emb = PositionalEncodingNEW(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(point_dim,2*context_dim,context_dim)
        # self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        # self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim)
        self.linear = ConcatSquashLinear(context_dim//2, point_dim, context_dim)
        #self.linear = nn.Linear(128,2)

        self.emb = nn.Conv1d(
            3,
            64,
            1,
            bias=True)

        self.demb = nn.Conv1d(
            64,
            3,
            1,
            bias=True)

        self.cnn_pred_1 = nn.Conv1d(2*context_dim, context_dim, 1,padding=0)
        self.cnn_pred_2 = nn.Conv1d(2*context_dim, context_dim//2, 3,padding=1)
        self.cnn_pred_3 = nn.Conv1d(2*context_dim, context_dim//8, 5,padding=2)
        self.cnn_pred_4 = nn.Conv1d(2*context_dim, context_dim//8, 7,padding=3)
        self.cnn_pred_5 = nn.Conv1d(2*context_dim, context_dim//8, 9,padding=4)
        self.cnn_pred_6 = nn.Conv1d(2*context_dim, context_dim//8, 11,padding=5)


    def forward(self, x, beta, context):
        # loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        # self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        batch_size = x.size(0) #256
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1) torch.Size([256, 1, 1])
        context = context.view(batch_size, 1, -1)   # (B, 1, F) torch.Size([256, 1, 256])

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=1)
        ctx_emb = context+time_emb
        
        embAttn = self.emb(ctx_emb)
        dembAttn = self.demb(embAttn)
        level_weight = F.softmax(dembAttn, dim=1)
        new_ctx_emb = level_weight * ctx_emb
        new_ctx_emb=torch.sum(new_ctx_emb,dim=1,keepdim=True)

        final_emb = self.concat1(new_ctx_emb,x) # torch.Size([256, 12, 512])
        final_emb = self.pos_emb(final_emb) 
        final_emb = final_emb.permute(0,2,1)

        out1=self.cnn_pred_1(final_emb) #torch.Size([256, 256, 12])
        out2=self.cnn_pred_2(final_emb) #torch.Size([256, 128, 12])
        out3=self.cnn_pred_3(final_emb) #torch.Size([256, 64, 12])
        out4=self.cnn_pred_4(final_emb) #torch.Size([256, 64, 12])
        out5=self.cnn_pred_5(final_emb) #torch.Size([256, 64, 12])
        out6=self.cnn_pred_6(final_emb) #torch.Size([256, 64, 12])
        trans=torch.concat((out1,out2,out3,out4,out5,out6),dim=1)  #torch.Size([256, 512, 12])


        trans = trans.permute(0,2,1)

       
        trans = self.concat3(new_ctx_emb, trans) # torch.Size([256, 12, 256])
        trans = self.concat4(new_ctx_emb, trans) # torch.Size([256, 12, 128])
        return self.linear(new_ctx_emb, trans) # torch.Size([256, 12, 2])

class TransformerConcatLinear(Module): # this

    def __init__(self, point_dim, config, context_dim, tf_layer, residual):
        # 2, 256, 3, False
        super().__init__()

        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(point_dim,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, point_dim, context_dim+3)
        #self.linear = nn.Linear(128,2)


    def forward(self, x, beta, context):
        # loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        # self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        batch_size = x.size(0) #256
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1) torch.Size([256, 1, 1])
        context = context.view(batch_size, 1, -1)   # (B, 1, F) torch.Size([256, 1, 256])

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3) torch.Size([256, 1, 3])
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3) torch.Size([256, 1, 259])
        x = self.concat1(ctx_emb,x) # torch.Size([256, 12, 512])
        final_emb = x.permute(1,0,2) # 12,256,512
        final_emb = self.pos_emb(final_emb) # 12,256,512


        trans = self.transformer_encoder(final_emb).permute(1,0,2) # torch.Size([256, 12, 512])
        trans = self.concat3(ctx_emb, trans) # torch.Size([256, 12, 256])
        trans = self.concat4(ctx_emb, trans) # torch.Size([256, 12, 128])
        return self.linear(ctx_emb, trans) # torch.Size([256, 12, 2])


class TransformerLinear(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim+3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
        #pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)







class LinearDecoder(Module):
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code):

        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out




class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.config= self.var_sched.config

        if(self.config.newLoss):
            self.cnn = nn.Conv1d(2, 5, 1, padding=0)

    def get_loss(self, x_0, context, t=None):
        # loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        loss2 = torch.tensor(0.)
        loss3 = torch.tensor(0.)
        if(self.config.newLoss): # 256,12,2
            x_0_o=x_0
            x_0=x_0.permute(0,2,1)
            x_0=self.cnn(x_0)
            x_0=x_0.permute(0,2,1)

            sx = torch.exp(x_0[:, :, 2])  # sx
            sy = torch.exp(x_0[:, :, 3])  # sy
            corr = torch.tanh(x_0[:, :, 4])  # corr

            cov = torch.zeros(x_0.shape[0], x_0.shape[1], 2, 2).to('cuda')
            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            cov[:, :, 1, 1] = sy * sy
            mean = x_0[:, :, 0:2]
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
            loss2 = - mvn.log_prob(x_0_o).sum()
            loss3 = F.mse_loss(mean.contiguous().view(-1, 2), x_0_o.contiguous().view(-1, 2), reduction='mean')


        batch_size, _, point_dim = x_0.size() #$ 256,12,2
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size) # 256 t 

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d) torch.Size([256, 12, 2])
        # print(e_rand.size())

        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context) # torch.Size([256, 12, 2])
        loss = F.mse_loss(e_theta.contiguous().view(-1, point_dim), e_rand.contiguous().view(-1, point_dim), reduction='mean')
        return [loss*self.config.loss1+loss2/self.config.loss2+loss3*self.config.loss3, loss ,loss2 ,loss3]

    def sample(self, num_points, context, sample, bestof, point_dim=2, flexibility=0.0, ret_traj=False):
        traj_list = []
        for i in range(sample):
            batch_size = context.size(0)
            if bestof: # torch.Size([16, 12, 2])
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T} # {100:xt}
            for t in range(self.var_sched.num_steps, 0, -1):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t]*batch_size]
                e_theta = self.net(x_t, beta=beta, context=context)
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        return torch.stack(traj_list)

    def sample_ddim(self, num_points, context, sample, bestof, point_dim=2, flexibility=0.0, ret_traj=False):
        traj_list = []
        self.alphas_cumprod = self.var_sched.alpha_bars
        for i in range(sample):
            batch_size = context.size(0)

            # model=self.diffusion.net
            # image_size=256
            # batch_size=256
            # channels=3
            ddim_timesteps=self.var_sched.num_ddim
            ddim_discr_method="uniform"
            ddim_eta=self.var_sched.config.ddim_eta
            clip_denoised=self.var_sched.config.clip_denoised

            c = self.var_sched.num_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.var_sched.num_steps, c)))
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            ddim_timestep_seq = ddim_timestep_seq + 1
            # previous sequence
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])


            if bestof: # torch.Size([16, 12, 2])
                sample_img = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                sample_img = torch.zeros([batch_size, num_points, point_dim]).to(context.device)

            for i in reversed(range(0, ddim_timesteps)) :
                t = torch.full((batch_size,), ddim_timestep_seq[i], device=context.device, dtype=torch.long)
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=context.device, dtype=torch.long)
                
                # 1. get current and previous alpha_cumprod
                
                alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
                alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
        
                # 2. predict noise using model
                beta = self.var_sched.betas[[t[0].item()]*batch_size]
                pred_noise = self.net(sample_img, beta=beta, context=context)
                
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
        return torch.stack(traj_list)

    def sample_ddim_new(self, num_points, context, sample, sample_out, bestof, point_dim=2, flexibility=0.0, ret_traj=False):
        traj_list = []
        self.alphas_cumprod = self.var_sched.alpha_bars
        for jj in range(sample_out):
            batch_size = context.size(0)

            # model=self.diffusion.net
            # image_size=256
            # batch_size=256
            # channels=3
            ddim_timesteps=self.var_sched.num_ddim
            ddim_discr_method="uniform"
            ddim_eta=self.var_sched.config.ddim_eta
            clip_denoised=self.var_sched.config.clip_denoised

            c = self.var_sched.num_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.var_sched.num_steps, c)))
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            ddim_timestep_seq = ddim_timestep_seq + 1
            # previous sequence
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])


            if bestof: # torch.Size([16, 12, 2])
                sample_img = torch.randn([batch_size, num_points, point_dim]).to(context.device)
                if(self.config.newLoss):
                    sample_img=sample_img.permute(0,2,1)
                    sample_img=self.cnn(sample_img)
                    sample_img=sample_img.permute(0,2,1)
            else:
                sample_img = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
                if(self.config.newLoss):
                    sample_img=sample_img.permute(0,2,1)
                    sample_img=self.cnn(sample_img)
                    sample_img=sample_img.permute(0,2,1)


            ddim_timesteps_test = ddim_timesteps
            # ddim_timesteps_test = self.config.ddim_timesteps_test
            for i in reversed(range(0, ddim_timesteps_test)) :
                t = torch.full((batch_size,), ddim_timestep_seq[i], device=context.device, dtype=torch.long)
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=context.device, dtype=torch.long)
                
                # 1. get current and previous alpha_cumprod
                
                alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
                alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
        
                # 2. predict noise using model
                beta = self.var_sched.betas[[t[0].item()]*batch_size]
                pred_noise = self.net(sample_img, beta=beta, context=context)
                
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

        return traj_list

    def sample_ddpm_new(self, num_points, context, sample, sample_out, bestof, point_dim=2, flexibility=0.0, ret_traj=False):
        traj_list = []
        for i in range(sample_out):
            batch_size = context.size(0)
            if bestof: # torch.Size([16, 12, 2])
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
                if(self.config.newLoss):
                    x_T=x_T.permute(0,2,1)
                    x_T=self.cnn(x_T)
                    x_T=x_T.permute(0,2,1)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
                if(self.config.newLoss):
                    x_T=x_T.permute(0,2,1)
                    x_T=self.cnn(x_T)
                    x_T=x_T.permute(0,2,1)
            traj = {self.var_sched.num_steps: x_T} # {100:xt}
            for t in range(self.var_sched.num_steps, 0, -1):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t]*batch_size]
                e_theta = self.net(x_t, beta=beta, context=context)
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        # return torch.stack(traj_list)

        return traj_list


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t).float()
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

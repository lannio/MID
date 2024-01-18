import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from .trajectory_utils import prediction_output_to_trajectories
#import visualization
from matplotlib import pyplot as plt
import pdb
import torch
import torch.distributions.multivariate_normal as torchdist
import torch.multiprocessing as multiprocessing
import numpy as np

def compute_ade(predicted_trajs, gt_traj):
    #pdb.set_trace()
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs

def compute_new_de(pred,gt,config):#256,12,5/2
    predlist=len(pred)
    kstep_V_pred_ls = []
    gt = gt.permute(1,0,2)*0.4
    for ii in range(predlist):
        predict=pred[ii].permute(1,0,2)
        sx = torch.exp(predict[:, :, 2])  # sx
        sy = torch.exp(predict[:, :, 3])  # sy
        corr = torch.tanh(predict[:, :, 4])  # corr

        cov = torch.zeros(predict.shape[0], predict.shape[1], 2, 2).to('cuda')
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = predict[:, :, 0:2]
        # dimensionality reminder: mean: [12, num_person, 2], cov: [12, num_person, 2, 2]

        """pytorch solution for sampling"""

        mvnormal = torchdist.MultivariateNormal(mean, cov)
        KSTEPS=config.sample
        for i in range(KSTEPS-1):
            kstep_V_pred_ls.append(torch.cumsum((mvnormal.sample()*0.4), dim=0))  # cat [12, num_person, 2]
        kstep_V_pred_ls.append(torch.cumsum(mean*0.4, dim=0))
    kstep_V_pred_ls = torch.stack(kstep_V_pred_ls, dim=0) # [KSTEPS, 12, num_person, 2]

    # kstep_V_pred = np.concatenate([traj for traj in kstep_V_pred_ls], axis=1) # [12, KSTEPS * num_person, 2]

    """end of sampling"""

    V_y_rel_to_abs =  torch.cumsum((gt), dim=0) # [12, num_person, 2] speed???)

    ade=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs),dim=3),dim=0)[0],dim=[0,1])
    fde=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs)[:,-1,:,:],dim=2),dim=0)[0],dim=[0])
    return [ade,fde]



def compute_new_de1010(pred,gt,config):#256,12,5/2

    pred=torch.stack(pred)

    gt = gt.permute(1,0,2)*0.4 #torch.Size([12, 256, 2])
    V_y_rel_to_abs =  torch.cumsum((gt), dim=0)

    meanpred=pred[:,:,:,0:2].permute(0,2,1,3)*0.4 #torch.Size([3, 12, 16, 2])
    V_y_pred_to_abs =  torch.cumsum((meanpred), dim=1)

    dis=torch.norm((V_y_pred_to_abs - V_y_rel_to_abs),dim=3)#torch.Size([3, 12, 16])
    index=torch.argmin(dis,dim=0)#torch.Size([12, 16])

    pred = pred.permute(2,1,3,0) # 12,256,5,3
    index=index.repeat(1,5,1,1).permute(2,3,1,0) # 12,16,5,1
    newpred=torch.gather(pred,3,index).squeeze(3) # torch.Size([12, 16, 5])

    # predlist=len(pred)
    kstep_V_pred_ls = []
    # gt = gt.permute(1,0,2)*0.4
    # for ii in range(predlist):
    predict=newpred
    sx = torch.exp(predict[:, :, 2])  # sx
    sy = torch.exp(predict[:, :, 3])  # sy
    corr = torch.tanh(predict[:, :, 4])  # corr

    cov = torch.zeros(predict.shape[0], predict.shape[1], 2, 2).to('cuda')
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = predict[:, :, 0:2]
    # dimensionality reminder: mean: [12, num_person, 2], cov: [12, num_person, 2, 2]

    """pytorch solution for sampling"""

    mvnormal = torchdist.MultivariateNormal(mean, cov)
    KSTEPS=config.sample
    for i in range(KSTEPS-1):
        kstep_V_pred_ls.append(torch.cumsum((mvnormal.sample()*0.4), dim=0))  # cat [12, num_person, 2]
    kstep_V_pred_ls.append(torch.cumsum(mean*0.4, dim=0))
    kstep_V_pred_ls = torch.stack(kstep_V_pred_ls, dim=0) # [KSTEPS, 12, num_person, 2]

    # kstep_V_pred = np.concatenate([traj for traj in kstep_V_pred_ls], axis=1) # [12, KSTEPS * num_person, 2]

    """end of sampling"""

    # V_y_rel_to_abs =  torch.cumsum((gt), dim=0) # [12, num_person, 2] speed???)

    ade=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs),dim=3),dim=0)[0],dim=[0,1])
    fde=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs)[:,-1,:,:],dim=2),dim=0)[0],dim=[0])
    return [ade,fde]

def compute_nogauss_de(pred,gt,config):#256,12,5/2
    predlist=len(pred)
    kstep_V_pred_ls = []
    gt = gt.permute(1,0,2)*0.4
    for ii in range(predlist):
        predict=pred[ii].permute(1,0,2)*0.4
        kstep_V_pred_ls.append(torch.cumsum(predict, dim=0))
    kstep_V_pred_ls = torch.stack(kstep_V_pred_ls, dim=0)
    V_y_rel_to_abs =  torch.cumsum((gt), dim=0) # [12, num_person, 2] speed???)

    ade=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs),dim=3),dim=0)[0],dim=[0,1])
    fde=torch.mean(torch.min(torch.norm((kstep_V_pred_ls - V_y_rel_to_abs)[:,-1,:,:],dim=2),dim=0)[0],dim=[0])
    return [ade,fde]


def compute_batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            #pdb.set_trace()
            #target_shape =
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])

    return batch_error_dict


# def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
#     for node_type in batch_errors_list[0].keys():
#         for metric in batch_errors_list[0][node_type].keys():
#             metric_batch_error = []
#             for batch_errors in batch_errors_list:
#                 metric_batch_error.extend(batch_errors[node_type][metric])

#             if len(metric_batch_error) > 0:
#                 log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

#                 if metric in bar_plot:
#                     pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

#                 if metric in box_plot:
#                     mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))


def batch_pcmd(prediction_output_dict,
               dt,
               max_hl,
               ph,
               node_type_enum,
               kde=True,
               obs=False,
               map=None,
               prune_ph_to_future=False,
               best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].append(np.array(ade_errors))
            batch_error_dict[node.type]['fde'].append(np.array(fde_errors))

    return batch_error_dict

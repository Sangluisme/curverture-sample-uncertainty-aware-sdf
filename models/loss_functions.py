# coding: utf-8

import torch
import torch.nn.functional as F
from third.diff_operators import *


def color_constraint_on_surf(gt_sdf, gt_color, pred_c):
    loss = torch.where(
        gt_sdf == 0.0,
        (gt_color - pred_c).abs(),
        torch.zeros_like(pred_c)
    )
    return loss

def color_constraint_off_surf(gt_sdf, gt_color, pred_c):
    loss = torch.where(
        (gt_sdf != 0.0) & (gt_sdf != -0.1),
        (pred_c - gt_color).abs(),
        torch.zeros_like(pred_c)
    )
    return loss


def weight_constraint_on_surf(gt_sdf, gt_weight, pred_w):
    loss = torch.where(
            gt_weight > 0,
            (gt_weight - pred_w)**2,
            torch.zeros_like(pred_w)
         )
        
    return loss



def weight_constraint_off_surf(gt_sdf, gt_weight, pred_w, radius=1e2):
    loss = torch.where(
        gt_sdf == -1,
        (pred_w)**2,
        torch.zeros_like(gt_weight)
        )
    return  loss



def sdf_constraint_on_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf == 0,
        (gt_sdf - pred_sdf)**2,
        torch.zeros_like(pred_sdf)
    )


def sdf_constraint_off_surf(gt_weight, gt_sdf, pred_sdf, radius=1e1):
    return torch.where(
        (gt_weight > 0) & (gt_sdf != 0),
        (gt_sdf - pred_sdf)**2,
        torch.zeros_like(pred_sdf)
        # torch.exp(-radius * torch.abs(pred_sdf))
    ) 


def vector_aligment_on_surf(gt_sdf, gt_vectors, pred_vectors):
    return torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None],
        torch.zeros_like(gt_sdf)
    )


def direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs):
    return torch.where(
        gt_sdf == 0,
        1 - (F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None])**2,
        torch.zeros_like(gt_sdf)
    )


def eikonal_constraint(grad):
    # half_size = grad.shape[1]//2
    loss = (grad.norm(dim=-1) - 1.) ** 2
    # loss[:,:half_size] = 0 * loss[:,:half_size]
    
    return loss

def normal_constraint(gt_sdf, gt_dirs, pred_dirs):
    
    return torch.where(
        gt_sdf == 0,
        (gt_dirs - pred_dirs).abs().norm(2, dim=1),
        torch.zeros_like(gt_sdf),
    )
    


def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
    """
    return torch.where(
           gt_sdf != -1,
           torch.zeros_like(pred_sdf),
           torch.exp(-radius * torch.abs(pred_sdf))
        )


def on_surface_normal_constraint(gt_sdf, gt_normals, grad):
    """
    This function return a number that measure how far gt_normals
    and grad are aligned in the zero-level set of sdf.
    """
    return torch.where(
           gt_sdf == 0,
           1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
           torch.zeros_like(grad[..., :1])
    )
    
def on_surf_normal_igr_loss(gt_sdf, gt_normals, grad):
    
    return torch.where(
           gt_sdf == 0,
           (gt_normals - grad).abs().norm(2, dim=1),
           torch.zeros_like(grad[..., :1])
    )


def weight_siren(model_output, gt, conf):
    """Uses true SDF value off surface and tries to fit the mean curvatures
    on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals', and
        'curvature' with the actual SDF values, the input data normals, and
        gaussian curvatures, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    
    lambda_sdf = conf.get('on_surf_sdf',3e3)
    lambda_surf = conf.get('off_surf_sdf', 1e2)
    lambda_wf = conf.get('off_surf_weight', 1e2)
    lambda_wo = conf.get('on_surf_weight', 3e3)
    lambda_e = conf.get('eikonal', 5e1)
    lambda_n = conf.get('normal_align', 1e2)
    
    
    gt_sdf = gt['sdf']
    gt_normals = gt['normals'].squeeze()
    gt_weight = gt["weights"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    pred_w = model_output['model_out'][...,-1]

    
    grad = gradient(pred_sdf, coords).squeeze()
    
    grad_loss = torch.where( (gt_sdf != 0.0),
                         ((grad.norm(2, dim=-1) - 1) ** 2),
                         torch.zeros_like(gt_sdf))
    
    
    sdf_on_surf = torch.where(
        (gt_sdf != -1) & (gt_weight > 0),
        (pred_sdf - gt_sdf).abs(),
        torch.zeros_like(pred_sdf)
    )
    
    sdf_off_surf = torch.where(
        (gt_sdf != -1),
        torch.zeros_like(gt_sdf),
        torch.exp(-1e2 * torch.abs(pred_sdf - gt_sdf))
        )
    
    
    weight_const_on_surf = weight_constraint_on_surf(gt_sdf, gt_weight, pred_w)
    weight_const_off_surf = weight_constraint_off_surf(gt_sdf, gt_weight, pred_w)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        # 'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * lambda_surf,
        # 'sdf_off_surf': sdf_constraint_off_surf(gt_weight, gt_sdf, pred_sdf).mean() * lambda_sdf,
        'sdf_on_surf': sdf_on_surf.mean() * lambda_surf,
        'sdf_off_surf': sdf_off_surf.mean() * lambda_sdf,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, grad).mean() * lambda_n ,#* 1e1,
        # 'grad_constraint': eikonal_constraint(grad).unsqueeze(-1).mean() * lambda_e,
        'grad_constraint': grad_loss.mean() * lambda_e,
        'weight_constraint_on': weight_const_on_surf.mean() * lambda_wo,
        'weight_constraint_off': weight_const_off_surf.mean() * lambda_wf,
        # 'curv_constraint': curv_constraint.mean() * 1e-1
    }
    

def weight_igr(model_output, gt, conf):
    """Uses true SDF value off surface and tries to fit the mean curvatures
    on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals', and
        'curvature' with the actual SDF values, the input data normals, and
        gaussian curvatures, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    
    lambda_sdf = conf.get('on_surf_sdf',3e3)
    lambda_surf = conf.get('off_surf_sdf', 1e2)
    lambda_wf = conf.get('off_surf_weight', 1e2)
    lambda_wo = conf.get('on_surf_weight', 3e3)
    lambda_e = conf.get('eikonal', 5e1)
    lambda_n = conf.get('normal_align', 1e2)
    
    
    gt_sdf = gt['sdf']
    gt_normals = gt['normals'].squeeze()
    gt_weight = gt["weights"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    pred_w = model_output['model_out'][...,-1]

    
    grad = gradient(pred_sdf, coords).squeeze()
    
    grad_loss = torch.where( (gt_sdf != 0.0),
                         ((grad.norm(2, dim=-1) - 1) ** 2),
                         torch.zeros_like(gt_sdf))
    
    
    sdf_on_surf = torch.where(
        (gt_sdf != -1) & (gt_weight > 0),
        (pred_sdf - gt_sdf).abs(),
        torch.zeros_like(pred_sdf)
    )
    
    # sdf_off_surf = torch.where(
    #     gt_sdf != -1, 
    #     torch.zeros_like(pred_sdf), 
    #     torch.exp(-1e2 * torch.abs(pred_sdf))
    #     )
    
    weight_const_on_surf = (pred_w - gt_weight)**2
    
    # weight_const_on_surf = weight_constraint_on_surf(gt_sdf, gt_weight, pred_w)
    # weight_const_off_surf = weight_constraint_off_surf(gt_sdf, gt_weight, pred_w)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_on_surf.mean() * lambda_surf,
        'sdf_off_surf': sdf_on_surf.mean() * lambda_surf,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, grad).mean() * lambda_n ,#* 1e1,
        # 'grad_constraint': eikonal_constraint(grad).unsqueeze(-1).mean() * lambda_e,
        'grad_constraint': grad_loss.mean() * lambda_e,
        'weight_constraint_on': weight_const_on_surf.mean() * lambda_wo,
        'weight_constraint_off': weight_const_on_surf.mean() * lambda_wf,
        # 'curv_constraint': curv_constraint.mean() * 1e-1
    }


def weight_color_sdf(model_output, gt):
    """Uses true SDF value off surface and tries to fit the mean curvatures
    on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals', and
        'curvature' with the actual SDF values, the input data normals, and
        gaussian curvatures, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_weight = gt["weights"]
    gt_color = gt['colors']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    pred_w = model_output['model_out'][...,1]
    pred_c = model_output['model_out'][...,-3:]

    grad = gradient(pred_sdf, coords).squeeze()
    
    weight_const_on_surf = weight_constraint_on_surf(gt_sdf, gt_weight, pred_w)
    weight_const_off_surf = weight_constraint_off_surf(gt_sdf, gt_weight, pred_w)

    color_on_surf = color_constraint_on_surf(gt_sdf, gt_color, pred_c)
    color_off_surf = color_constraint_off_surf(gt_sdf, gt_color, pred_c)
    
    color_r = color_on_surf[...,0].mean()
    color_g = color_on_surf[...,1].mean()
    color_b = color_on_surf[...,2].mean()
    
    color_on_surf_loss = color_b + color_r +color_g
    
    color_off_r = color_off_surf[...,0].mean()
    color_off_g = color_off_surf[...,1].mean()
    color_off_b = color_off_surf[...,2].mean()
    
    color_off_surf_loss = color_off_r + color_off_g + color_off_b
    
    

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        'sdf_off_surf': sdf_constraint_off_surf(gt_weight, gt_sdf, pred_sdf).mean() * 2e2,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, grad).mean() *1e2 ,#* 1e1,
        'grad_constraint': eikonal_constraint(grad).unsqueeze(-1).mean() * 5e1,
        'weight_constraint_on': weight_const_on_surf.mean() * 3e2,
       'weight_constraint_off': weight_const_off_surf.mean() * 3e2,
        'color_on_surf': color_on_surf_loss * 5e4,
        'color_off_surf': color_off_surf_loss * 1e4,
    }


def siren_loss(model_output, gt, conf):
    
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_weight = gt["weights"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    pred_w = model_output['model_out'][...,-1]

    
    grad = gradient(pred_sdf, coords)
    
    sdf_constraint = torch.where(
        gt_sdf != -1,
        pred_sdf.abs(), 
        torch.zeros_like(pred_sdf)
    )
    
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)
    
    return{
        'sdf_on_surf': sdf_constraint.mean() * 3e3,
        'sdf_off_surf': off_surface_without_sdf_constraint(gt_sdf, pred_sdf).mean() * 1e2,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, grad).mean() * 1e2 ,#* 1e1,
        'grad_constraint': grad_constraint.mean() * 5e1,
    }

def IGR_loss(model_output, gt, conf):
    gt_sdf = gt['sdf'][...,0]
    gt_normals = gt['normals'].squeeze()
    gt_weight = gt["weights"][...,0]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    pred_w = model_output['model_out'][...,-1]

    
    grad = gradient(pred_sdf, coords).squeeze()
    
    lambda_sdf = conf.get('on_surf_sdf',3e2)
    lambda_surf = conf.get('off_surf_sdf', 3e1)
    lambda_e = conf.get('eikonal', 3e2)
    lambda_n = conf.get('normal_align', 3e2)
    
    mnfld_loss = torch.where( gt_sdf == 0, 
                             (gt_sdf - pred_sdf).abs(),
                             torch.zeros_like(gt_sdf))
    
    grad_loss = torch.where( gt_sdf == -1,
                         ((grad.norm(2, dim=-1) - 1) ** 2),
                         torch.zeros_like(gt_sdf))
    
    return{
        'sdf_on_surf': mnfld_loss.mean() * lambda_surf,
        # 'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * lambda_surf,
        'sdf_off_surf': off_surface_without_sdf_constraint(gt_sdf, pred_sdf).mean() * 0.0,
        'normal_constraint': normal_constraint(gt_sdf, gt_normals, grad).mean() * lambda_n ,#* 1e1,
        'grad_constraint': grad_loss.mean() * lambda_e,
    #     'weight_constraint_on': 0,
    #    'weight_constraint_off': 0,
    }


def weight_neural_pull(model_output, gt, conf):

    lambda_w = conf.get('on_surf_weight',3e1)
    lambda_surf = conf.get('off_surf_sdf', 3e2)
    

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    pred_w = model_output['model_out'][...,-1]

    gt_weight = gt["weights"][...,0]
    points = gt['points'].squeeze()

    grad = gradient(pred_sdf, coords)
    
    grad_norm = F.normalize(grad, dim=2)             # 5000x3
    sample_moved = coords - grad_norm * pred_sdf.unsqueeze(-1)                 # 5000x3

    loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1)

    loss_weight = (gt_weight - pred_w)**2
    
    # grad_loss = torch.abs(grad.norm(dim=-1) - 1)



    return {'sdf_on_surf': loss_sdf.mean() * lambda_surf,
        'weight_loss': loss_weight.mean() * lambda_w,
        # 'grad_constraint': grad_loss.mean() * lambda_surf,
    }


def neural_pull_loss(model_output, gt, conf):

    # lambda_w = conf.get('on_surf_weight',3e1)
    lambda_surf = conf.get('off_surf_sdf', 3e2)
    

    coords = model_output['model_in']
    pred_sdf = model_output['model_out'][...,0]
    # pred_w = model_output['model_out'][...,-1]

    # gt_weight = gt["weights"][...,0]
    points = gt['points'].squeeze()

    grad = gradient(pred_sdf, coords).squeeze()
    
    grad_norm = F.normalize(grad, dim=1)                # 5000x3
    sample_moved = coords - grad_norm * pred_sdf                 # 5000x3

    loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1)

    # loss_weight = (gt_weight - pred_w)**2
    
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)



    return {'sdf_on_surf': loss_sdf.mean() * lambda_surf,
        'weight_loss': 0.0
    }

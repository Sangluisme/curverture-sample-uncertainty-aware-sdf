
import torch
import torch.nn.functional as F
from third.diff_operators import gradient


def cauchy_estimator(l, loss):
    l2 = l**2
    
    return 0.5*l2*torch.log(1 + loss / l2)

# normal loss
def vector_aligment_on_surf(gt_sdf, normals, pred_grad):
    return torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(pred_grad, normals, dim=-1)[..., None],
        torch.zeros_like(gt_sdf)
    ).mean()


def direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs):
    return torch.where(
        gt_sdf == 0,
        1 - (F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None])**2,
        torch.zeros_like(gt_sdf)
    ).mean()
    
    
def normal_loss(gt_sdf, pred_grad, normals):
    loss = torch.where(
        gt_sdf == 0,
        ((pred_grad - normals).abs()).norm(2, dim=1),
        torch.zeros_like(gt_sdf),
     )
    return loss.mean()

# sdf loss
def sdf_loss(gt_sdf, pred_sdf, gt_w=None):
    
    loss = (gt_sdf - pred_sdf).abs()
    
    return loss.mean()

def weight_sdf_loss(gt_sdf, pred_sdf, gt_w=None):
    
    loss = torch.where(
        gt_w > 0,
        (gt_sdf - pred_sdf).abs(),
        torch.zeros_like(gt_sdf)
        )

    return loss.mean()

def siren_sdf_loss(gt_sdf, pred_sdf, gt_w=None):
    
    sdf_on_surf = torch.where(
        (gt_sdf != -1) & (gt_w > 0),
        (gt_sdf - pred_sdf).abs(),
        torch.zeros_like(gt_sdf),
        )
    
    sdf_off_surf = torch.where(
        (gt_sdf != -1),
        torch.zeros_like(gt_sdf),
        torch.exp(-1e2 * torch.abs(pred_sdf - gt_sdf))
        )
    
    return 0.1*sdf_off_surf.mean() + sdf_on_surf.mean()

def weight_siren_sdf_loss(gt_sdf, pred_sdf, gt_w=None):
    
    sdf_on_surf = torch.where(
        (gt_sdf != -1),
        (gt_sdf - pred_sdf).abs(),
        torch.zeros_like(gt_sdf),
        )
    
    sdf_off_surf = torch.where(
        (gt_sdf != -1),
        torch.zeros_like(gt_sdf),
        torch.exp(-1e3 * torch.abs(pred_sdf - gt_sdf))
        )
    
    return 0.1*sdf_off_surf.mean() + sdf_on_surf.mean()
    
# weight loss
def weight_loss(gt_w, pred_w):
    
    # gt = torch.where(
    #     gt_w > 0,
    #     torch.ones_like(gt_w),
    #     torch.zeros_like(gt_w)
    #     )
    
    loss = (pred_w - gt_w)**2

    loss = loss.mean()
    
    return loss


def igr_eikonal_loss(grad):
    return ((grad.norm(2, dim=-1) - 1) ** 2).mean()

def siren_eikonal_loss(grad):
    return torch.abs(grad.norm(dim=-1) - 1).mean()


# neuralpull loss
def neuralpull(samples, points, sdf, grad):
    grad_norm = F.normalize(grad, dim=1)   
    sample_moved = samples - grad_norm * sdf.unsqueeze(-1)
    loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1)
    l = 0.01
    
    # loss = cauchy_estimator(l, loss_sdf)
    
    return loss_sdf.mean()

def weight_neuralpull(samples, points, sdf, grad, gt_w):
    grad_norm = F.normalize(grad, dim=1)   
    sample_moved = samples - grad_norm * sdf.unsqueeze(-1)
    loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1)
    
    # loss = torch.where(
    #     gt_w > 0,
    #     loss_sdf,
    #     torch.zeros_like(sdf)
    # )
    
    return loss_sdf.mean()


class Loss:
    
    def __init__(self, conf) -> None:
        self.conf = conf
        
        self.l_sdf = conf.get('sdf',1)
        self.l_off_sdf = conf.get('off_surf_sdf', 1)
        self.l_w = conf.get('weight', 1)
        self.l_e = conf.get('eikonal', 0.1)
        self.l_n = conf.get('normal', 1)
        
        self.mode = conf['type']
        
    def get_loss(self):
        
        self.weight_loss = None
        
        if self.mode == 'igr' or self.mode == 'IGR':
            
            self.normal_loss = normal_loss
            self.sdf_loss = sdf_loss
            self.eikonal_loss = igr_eikonal_loss
        
        elif self.mode == 'siren' or self.mode == 'SIREN':
            
            self.normal_loss = vector_aligment_on_surf
            self.sdf_loss = siren_sdf_loss
            self.eikonal_loss = siren_eikonal_loss
        
        elif self.mode =='weight_igr':
            
            self.normal_loss = normal_loss
            self.sdf_loss = weight_sdf_loss
            self.weight_loss = weight_loss
            self.eikonal_loss = igr_eikonal_loss

            
        elif self.mode == 'weight_siren':
            
            self.normal_loss = vector_aligment_on_surf
            self.sdf_loss = weight_siren_sdf_loss
            self.weight_loss = weight_loss
            self.eikonal_loss = siren_eikonal_loss

        elif self.mode == 'neural_pull':
            self.sdf_loss = weight_neuralpull
            self.normal_loss = None
            self.eikonal_loss = None
            
        elif self.mode == 'weight_neural_pull':
            self.sdf_loss = weight_neuralpull
            self.weight_loss = weight_loss
            self.eikonal_loss = None
            self.normal_loss = None
        
        else:
            assert("Unknown loss option.")
            raise NotImplementedError

        print(self.sdf_loss, self.normal_loss, self.weight_loss)
            
            
            
    def compute_loss(self, model_in, model_out, gt):
    
        
        pred_sdf = model_out[:,0]
        if model_out.shape[1] > 1:
            pred_w = model_out[:,-1]
            
        if 'gt_sdf' in gt:
            gt_sdf = gt['gt_sdf']
        else:
            gt_sdf = torch.zeros_like(pred_sdf)
        
        if 'gt_w' in gt:
            gt_w = gt['gt_w']
            
        else:
            gt_w = torch.ones_like(pred_sdf)
            
        grad = gradient(model_out, model_in)
        
        if self.mode == "neural_pull" or self.mode == 'weight_neural_pull':
            points = gt['points']
            # gt_w = torch.ones_like(pred_sdf)
            sdf = self.sdf_loss(samples=model_in, points=points, sdf=pred_sdf, grad=grad, gt_w=gt_w)
        else:
            sdf = self.sdf_loss(gt_sdf=gt_sdf, pred_sdf=pred_sdf, gt_w=gt_w)
        
        loss = self.l_sdf * sdf
        
        loss_dic = {
            'Manifold loss': sdf,
        }
        
        if 'gt_normals' in gt:
            gt_normal = gt['gt_normals']
            normal = self.normal_loss(gt_sdf=gt_sdf, pred_grad=grad, normals=gt_normal)
            loss = self.l_n * normal + loss
            loss_dic.update({'Normals loss': normal})
 
        if self.weight_loss is not None:
            
            weight = self.weight_loss(gt_w, pred_w)
            loss = self.l_w * weight + loss
            loss_dic.update({'Weight loss': weight})
        
        if self.eikonal_loss is not None:
            # compute eikonal loss
            loss_e = self.eikonal_loss(grad)
            loss_dic.update({'eikonal loss': loss_e})
        
            loss = self.l_e * loss_e + loss
            
        return loss, loss_dic
    
    
    def compute_eikonal_loss(self, model_in, model_out):
        
        # pred_sdf = model_out[:,0]
        
        grad = gradient(model_out, model_in)
        
        eikonal = self.eikonal_loss(grad)
        
        return self.l_e * eikonal
        
        
            
        
        
        
            
        


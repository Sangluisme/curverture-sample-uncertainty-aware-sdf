import cv2 as cv
import numpy as np
import torch.nn as nn
import torch

class DifferentialGeometryEstimator:
    def __init__(
        self,
        K,
        img_res,
        window_size=5
        ):
        
        self.img_res_ = img_res
        self.window_size = window_size 
        self.cx = K[0,2]
        self.cy = K[1,2]
        self.fx = K[0,0]
        self.fy = K[1,1]

        self.cache()

    def pixelgrid(self):
        u, v = np.mgrid[0:self.img_res_[1], 0:self.img_res_[0]].astype(np.float32)
        u = u.T - self.cx
        v = v.T - self.cy

        return u, v

    def cache(self):
        fx_inv = 1./self.fx
        fy_inv = 1./self.fy
        
        x0, y0 = self.pixelgrid()

        x0 = fx_inv * x0
        x0_sq = x0 * x0
        y0 = fy_inv * y0
        y0_sq = y0 * y0
        x0_y0 = x0 * y0

        n_sq = 1.0 + x0_sq + y0_sq
        n_sq_inv = 1 / n_sq
        x0_n_sq_inv = x0 * n_sq_inv
        y0_n_sq_inv = y0 * n_sq_inv
        
        self.x0 = x0
        self.y0 = y0
        
    def covert_pointcloud(self, depth):
        x = self.x0 * depth
        y = self.y0 * depth
        
        pc = np.stack((x,y,depth), axis=-1)
        
        return pc 
    
    def compute_first_fundamental_form(self, depth):
        
        point_cloud = self.covert_pointcloud(depth)
        
        dP_dx = cv.Sobel(point_cloud, cv.CV_64F, 1, 0, ksize = self.window_size)
        dP_dy = cv.Sobel(point_cloud, cv.CV_64F, 0, 1, ksize = self.window_size)
        
        # Calculate the First Fundamental Form matrix
        I_11 = np.sum(dP_dx * dP_dx, axis=-1)
        I_12 = np.sum(dP_dx * dP_dy, axis=-1)
        I_22 = np.sum(dP_dy * dP_dy, axis=-1)
        
        return I_11, I_12, I_22
    
    def compute_second_fundamental_form(self, N):
        normal = np.stack((N[0,:,:],N[1,:,:],N[2,:,:]),axis=-1)
        
        dN_dx = cv.Sobel(normal, cv.CV_64F, 1, 0, ksize = self.window_size)
        dN_dy = cv.Sobel(normal, cv.CV_64F, 0, 1, ksize = self.window_size)
        
        # Calculate the Second Fundamental Form (shape operator) matrix
        II_11 = np.sum(dN_dx * dN_dx, axis=-1)
        II_12 = np.sum(dN_dx * dN_dy, axis=-1)
        II_22 = np.sum(dN_dy * dN_dy, axis=-1)
        
        return II_11, II_12, II_22
        
        
        
    def derivatives(self, img):
        
        dy = cv.Sobel(img, cv.CV_64F, 1, 0, ksize = self.window_size)
        ddy = cv.Sobel(dy, cv.CV_64F, 1, 0, ksize = self.window_size)
        
        dx = cv.Sobel(img, cv.CV_64F, 0, 1, ksize = self.window_size)
        ddx = cv.Sobel(dx, cv.CV_64F, 0, 1, ksize = self.window_size)
        
        dxy = cv.Sobel(dx, cv.CV_64F, 1, 0, ksize = self.window_size)
        
        # laplacian = cv.Laplacian(img,cv.CV_64F)
        
        return dy, ddy, dx, ddx, dxy
    
    
    def compute(self, depth):
        
        dy, ddy, dx, ddx, dxy = self.derivatives(depth)
        
        denom = 1 + dx * dx + dy * dy
        
        K = ddx * ddy - dxy * dxy
        
        K = K / (denom * denom)
        
        H = (1 + dy * dy) * ddx - 2 * dx * dy * dxy + (1 + dx * dx) * ddy
        
        H_donom = 2 * np.sqrt(denom * denom * denom)
        
        H = H / H_donom
        
        H = np.where(H < -10, 0, H)
        H = np.where(H > 10, 0, H)
        
        K = np.where(K < -10, 0, K)
        K = np.where(K > 10, 0, K)
        
        Cest = {
            'gaussian_curv':K,
            'mean_curv': H,
            # 'laplacian': laplacian
        }
        
        return Cest
    
    
    def compute_from_fundamental_form(self, depth, normal):
     
        
        E, F, G = self.compute_first_fundamental_form(depth)
        L, M, N = self.compute_second_fundamental_form(normal)
        
        K = L * N - M * M 
        
        denom_K  = E * G - F * F 
        
        H = E * N + G * L - 2 * F * M
        denom_H = 2 * denom_K
        
        H = H / denom_H
        H = np.where(denom_K == 0, 0, H)
        
        K = K / denom_K
        K = np.where(denom_K == 0, 0, K)

        # clip data incase some are to large and destory the whole estimation
        H = np.where(H < -20, 0, H)
        H = np.where(H > 20, 0, H)
        
        K = np.where(K < -20, 0, K)
        K = np.where(K > 20, 0, K)
        
        Cest = {
            'gaussian_curv':K,
            'mean_curv': H,
            # 'laplacian': laplacian
        }
        
        return Cest

        
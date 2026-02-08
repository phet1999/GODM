import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error as MSE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset, random_split,Dataset
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from scipy.special import digamma
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
from functorch import make_functional, jvp, grad, hessian, jacfwd, jacrev,make_functional_with_buffers
import statsmodels.api as sm
import warnings
from torch.autograd.functional import jacobian
import copy
from scipy.signal import periodogram
import time
import torch.nn.functional as FF
from functorch import make_functional, vmap, vjp, jvp, jacrev
import numbers
from torch.nn import init
import lightning as L
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from typing import Literal, Optional
from sklearn.preprocessing import StandardScaler
import os
_has_sklearn = True
from numba import njit, prange
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_data_fast(data: pd.DataFrame, target_cols: list, look_back: int, look_ahead: int):

    values = data[target_cols].values

    num_samples = len(values) - look_back - look_ahead + 1
    num_features = len(target_cols)

    if num_samples <= 0:
        return None, None

   
    stride = values.strides[0] 
    itemsize = values.itemsize   

    X_view = np.lib.stride_tricks.as_strided(
        values,
        shape=(num_samples, look_back, num_features),
        strides=(stride, stride, itemsize)
    )

    Y_view = np.lib.stride_tricks.as_strided(
        values[look_back:], 
        shape=(num_samples, look_ahead, num_features),
        strides=(stride, stride, itemsize)
    )

    X_tensor = torch.from_numpy(X_view.copy()).float()
    Y_tensor = torch.from_numpy(Y_view.copy()).float()

    return X_tensor, Y_tensor

class MyDataset(Dataset):
    def __init__(self, X_his,y):
        
        self.X_his = torch.tensor(X_his, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        

    def __len__(self):
        return len(self.X_his)

    def __getitem__(self, idx):
        
        return self.X_his[idx],self.y[idx]



class SpatioTemporalDataModule(L.LightningDataModule):
    def __init__(self, train_x, train_c, val_x,val_c,test_x, test_c, batch_size,num_workers=0):
        super().__init__()
        self.train_dataset = TensorDataset(train_x, train_c)
        self.val_dataset = TensorDataset(val_x, val_c)
        self.test_dataset = TensorDataset(test_x, test_c)
        self.batch_size = batch_size
        self.num_workers = num_workers 
    def _create_dict_dataloader(self, dataset, shuffle):
        class DictDataLoader(DataLoader):
            def __iter__(self):
                for x, c in super().__iter__():
                    yield {"x": x, "c": c}
        return DictDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers, 
            persistent_workers=True if self.num_workers > 0 else False 
        )
    def _create_predict_dict_dataloader(self, dataset, shuffle):
        class DictDataLoader(DataLoader):
            def __iter__(self):
                for x, c in super().__iter__():
                    yield {"x": x, "c": c}
        return DictDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False 
        )
    def train_dataloader(self): return self._create_dict_dataloader(self.train_dataset, shuffle=True)
    def val_dataloader(self): return self._create_dict_dataloader(self.val_dataset, shuffle=False)
    def predict_dataloader(self): return self._create_predict_dict_dataloader(self.test_dataset,shuffle=False)


def split_and_scale_data(df: pd.DataFrame, train_split: float = 0.8, val_split: float = 0.9):
    
    len_df = len(df)
    train_end = int(train_split * len_df)
    val_end = int(val_split * len_df)
  
    train = df[:train_end]
    val = df[train_end:val_end]
    test = df[val_end:]
 
    scaler = StandardScaler()
 
    train_scaled = scaler.fit_transform(train.values)
    train_scaled_df = pd.DataFrame(train_scaled, columns=train.columns, index=train.index)
 
    if not val.empty:
        val_scaled = scaler.transform(val.values)
        val_scaled_df = pd.DataFrame(val_scaled, columns=val.columns, index=val.index)
    else:
        val_scaled_df = pd.DataFrame(columns=df.columns) 
 
    if not test.empty:
        test_scaled = scaler.transform(test.values)
        test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns, index=test.index)
    else:
        test_scaled_df = pd.DataFrame(columns=df.columns) 
        
    return train_scaled_df, val_scaled_df, test_scaled_df, scaler


def normalize_adjacency_matrix_efficient(adj, add_self_loops: bool = True):
    if not isinstance(adj, torch.Tensor):
        adj_tensor = torch.FloatTensor(adj.copy())
    else:
        adj_tensor = adj.clone()

    if add_self_loops:
        adj_tensor.fill_diagonal_(1)

    degree = torch.sum(adj_tensor, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    adj_tensor = adj_tensor * d_inv_sqrt.unsqueeze(1)

    norm_adj_matrix = adj_tensor * d_inv_sqrt.unsqueeze(0)

    return adj,norm_adj_matrix



def generate_ortho_matrices(
    train_data: np.ndarray, 
    seq_len: int, 
    pred_len: int, 
    q_chan_indep: bool = True, 
    stride: int = 1,
    save_dir: str = './dataset/'
):
    
    # 数据校验
    assert train_data.ndim == 2, "train_data shape usually should be (Time, Channels)"
    L, N = train_data.shape
    
    print(f"Start generating Ortho Matrices...")
    print(f"Data shape: {train_data.shape}, Seq_len: {seq_len}, Pred_len: {pred_len}")
    print(f"Channel Independence: {q_chan_indep}")

    corr_accum_in = []
    corr_accum_out = []
    
    
    num_samples = (L - seq_len - pred_len) // stride + 1
    if num_samples <= 0:
        raise ValueError("Data length is too short for the given seq_len and pred_len.")

    for n in range(N):
        channel_data = train_data[:, n]
        
        shape_in = (seq_len, num_samples)
        strides_in = (channel_data.strides[0], channel_data.strides[0] * stride)
        windows_in = np.lib.stride_tricks.as_strided(
            channel_data, shape=shape_in, strides=strides_in
        ) # Shape: [T, Samples]
        
        # Output part (pred_len)
        # Offset by seq_len
        offset_data = channel_data[seq_len:]
        shape_out = (pred_len, num_samples)
        strides_out = (offset_data.strides[0], offset_data.strides[0] * stride)
        windows_out = np.lib.stride_tricks.as_strided(
            offset_data, shape=shape_out, strides=strides_out
        ) # Shape: [tau, Samples]

        coef_in = np.corrcoef(windows_in)
        
        coef_out = np.corrcoef(windows_out)
        
        coef_in = np.nan_to_num(coef_in, nan=0.0)
        coef_out = np.nan_to_num(coef_out, nan=0.0)

        corr_accum_in.append(coef_in)
        corr_accum_out.append(coef_out)

    corr_tensor_in = np.stack(corr_accum_in, axis=0)
    corr_tensor_out = np.stack(corr_accum_out, axis=0)
    
    
    def get_eigen_vectors(corr_matrix):
       
        _, eigen_vecs = np.linalg.eigh(corr_matrix)
       
        return eigen_vecs

    if q_chan_indep:
        # Input shape: [N, seq_len, seq_len] -> Output: [N, seq_len, seq_len]
        Q_mat = get_eigen_vectors(corr_tensor_in)
        # Input shape: [N, pred_len, pred_len] -> Output: [N, pred_len, pred_len]
        Q_out_mat = get_eigen_vectors(corr_tensor_out)
        
        suffix = "indep"
    else:
        # Process: Mean over N -> [seq_len, seq_len] -> Eigen -> [seq_len, seq_len]
        corr_mean_in = np.mean(corr_tensor_in, axis=0)
        corr_mean_out = np.mean(corr_tensor_out, axis=0)
        
        Q_mat = get_eigen_vectors(corr_mean_in)
        Q_out_mat = get_eigen_vectors(corr_mean_out)
        
        suffix = "shared"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    name_in = f"Q_mat_sl{seq_len}_{suffix}.npy"
    name_out = f"Q_out_mat_pl{pred_len}_{suffix}.npy"
    
    path_in = os.path.join(save_dir, name_in)
    path_out = os.path.join(save_dir, name_out)
    
    np.save(path_in, Q_mat.astype(np.float32))
    np.save(path_out, Q_out_mat.astype(np.float32))
    
    print(f"Done.")
    print(f"Q_mat shape: {Q_mat.shape}, Saved to: {path_in}")
    print(f"Q_out_mat shape: {Q_out_mat.shape}, Saved to: {path_out}")
    
    return path_in, path_out








def calculate_crps(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    if predictions.shape[1:] != ground_truth.shape:
        raise ValueError(
            f"Predictions shape (excluding sample dim) {predictions.shape[1:]} "
            f"must match ground truth shape {ground_truth.shape}"
        )
    
    num_samples = predictions.shape[0]
    
    abs_error = np.abs(predictions - ground_truth)
    
    accuracy_term = np.mean(abs_error, axis=0)
 
    sorted_preds = np.sort(predictions, axis=0)
    
    m = num_samples
    j = np.arange(m)
    coefficients = 2 * j - m + 1

    coeff_shape = (m,) + (1,) * (predictions.ndim - 1)
    coefficients = coefficients.reshape(coeff_shape)
 
    weighted_sum = coefficients * sorted_preds
    
    sum_weighted_sum = np.sum(weighted_sum, axis=0)
    dispersion_term = sum_weighted_sum / (m ** 2)
 
    cprs_per_point = accuracy_term - dispersion_term
    
    final_crps_score = np.mean(cprs_per_point)
    
    return float(final_crps_score)


import numpy as np
import os

def generate_ortho_matrices_lambda(   
    train_data: np.ndarray,   
    seq_len: int,   
    pred_len: int,   
    q_chan_indep: bool = True,   
    stride: int = 1,   
    save_dir: str = './dataset/'   
):    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
      
    assert train_data.ndim == 2, "train_data shape usually should be (Time, Channels)"   
    L, N = train_data.shape  
      
    print(f"Start generating Ortho Matrices and Eigenvalues...")   
    print(f"Data shape: {train_data.shape}, Seq_len: {seq_len}, Pred_len: {pred_len}")   
    print(f"Channel Independence: {q_chan_indep}")   
  
    corr_accum_in = []   
    corr_accum_out = []   
      
    num_samples = (L - seq_len - pred_len) // stride + 1  
    if num_samples <= 0:   
        raise ValueError("Data length is too short for the given seq_len and pred_len.")   
  
    for n in range(N):   
        channel_data = train_data[:, n]   
          
        # Input part
        shape_in = (seq_len, num_samples)   
        strides_in = (channel_data.strides[0], channel_data.strides[0] * stride)   
        windows_in = np.lib.stride_tricks.as_strided(channel_data, shape=shape_in, strides=strides_in)   
          
        # Output part
        offset_data = channel_data[seq_len:]   
        shape_out = (pred_len, num_samples)   
        strides_out = (offset_data.strides[0], offset_data.strides[0] * stride)   
        windows_out = np.lib.stride_tricks.as_strided(offset_data, shape=shape_out, strides=strides_out)   
  
        coef_in = np.corrcoef(windows_in)   
        coef_out = np.corrcoef(windows_out)   
          
        coef_in = np.nan_to_num(coef_in, nan=0.0)   
        coef_out = np.nan_to_num(coef_out, nan=0.0)   
  
        corr_accum_in.append(coef_in)   
        corr_accum_out.append(coef_out)   
  
    corr_tensor_in = np.stack(corr_accum_in, axis=0)   
    corr_tensor_out = np.stack(corr_accum_out, axis=0)   
      
    def get_eigen_decomposition(corr_matrix):   
       
       
        w, v = np.linalg.eigh(corr_matrix)   
        
       
        if w.ndim == 1: # Shared mode [T]
            w = w[::-1]
            v = v[:, ::-1]
        else: # Indep mode [N, T]
            w = w[:, ::-1]
            v = v[:, :, ::-1]
            
        return w, v  
  
    if q_chan_indep:   
        # [N, T, T] -> w: [N, T], v: [N, T, T]
        Lambda_mat, Q_mat = get_eigen_decomposition(corr_tensor_in)   
        Lambda_out_mat, Q_out_mat = get_eigen_decomposition(corr_tensor_out)   
        suffix = "indep"   
    else:   
        corr_mean_in = np.mean(corr_tensor_in, axis=0)   
        corr_mean_out = np.mean(corr_tensor_out, axis=0)   
        # [T, T] -> w: [T], v: [T, T]
        Lambda_mat, Q_mat = get_eigen_decomposition(corr_mean_in)   
        Lambda_out_mat, Q_out_mat = get_eigen_decomposition(corr_mean_out)   
        suffix = "shared"   
  
    if not os.path.exists(save_dir):   
        os.makedirs(save_dir)   
          
    path_in = os.path.join(save_dir, f"Q_mat_sl{seq_len}_{suffix}.npy")   
    path_out = os.path.join(save_dir, f"Q_out_mat_pl{pred_len}_{suffix}.npy")   
    path_lambda = os.path.join(save_dir, f"Lambda_sl{seq_len}_{suffix}.npy")
    path_lambda_out = os.path.join(save_dir, f"Lambda_pl{pred_len}_{suffix}.npy")
      
    np.save(path_in, Q_mat.astype(np.float32))   
    np.save(path_out, Q_out_mat.astype(np.float32))   
    np.save(path_lambda, Lambda_mat.astype(np.float32))
    np.save(path_lambda_out, Lambda_out_mat.astype(np.float32))
      
    print(f"Done.")   
    print(f"Q_mat shape: {Q_mat.shape}, Lambda shape: {Lambda_mat.shape}")   
    print(f"Files saved to: {save_dir}")   
      
    return path_in, path_out, path_lambda, path_lambda_out



from scipy.fftpack import dct

def generate_ortho_matrices_lambda_DCT(
    train_data: np.ndarray,
    seq_len: int,
    pred_len: int,
    q_chan_indep: bool = True,
    stride: int = 1,
    save_dir: str = './dataset/'
):
  
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    L, N = train_data.shape
    print(f"Generating Frequency (DCT) Matrices...")
    
   
    def get_dct_matrix(dim):
    
        I = np.eye(dim)
     
        Q = dct(I, axis=0, norm='ortho')
        return Q # [dim, dim]

    Q_in_fixed = get_dct_matrix(seq_len)
    Q_out_fixed = get_dct_matrix(pred_len)

    
    def compute_power(data, Q, length):
        # data: [L]
        num_samples = (len(data) - length - pred_len) // stride + 1
        shape = (length, num_samples)
        strides = (data.strides[0], data.strides[0] * stride)
        windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
       
        spec = Q @ windows 
       
        power = np.mean(spec**2, axis=1)
        return power

    all_lambda_in = []
    all_lambda_out = []

    for n in range(N):
        channel_data = train_data[:, n]
        
        
        l_in = compute_power(channel_data, Q_in_fixed, seq_len)
      
        l_out = compute_power(channel_data[seq_len:], Q_out_fixed, pred_len)
        
        all_lambda_in.append(l_in)
        all_lambda_out.append(l_out)

   
    if q_chan_indep:
       
        Lambda_mat = np.stack(all_lambda_in, axis=0)      # [N, seq_len]
        Lambda_out_mat = np.stack(all_lambda_out, axis=0)  # [N, pred_len]
       
        Q_mat = np.tile(Q_in_fixed[np.newaxis, :, :], (N, 1, 1))
        Q_out_mat = np.tile(Q_out_fixed[np.newaxis, :, :], (N, 1, 1))
        suffix = "indep"
    else:
       
        Lambda_mat = np.mean(np.stack(all_lambda_in, axis=0), axis=0) # [seq_len]
        Lambda_out_mat = np.mean(np.stack(all_lambda_out, axis=0), axis=0) # [pred_len]
        Q_mat = Q_in_fixed
        Q_out_mat = Q_out_fixed
        suffix = "shared"

    path_in = os.path.join(save_dir, f"Q_mat_sl{seq_len}_{suffix}.npy")
    path_out = os.path.join(save_dir, f"Q_out_mat_pl{pred_len}_{suffix}.npy")
    path_lambda = os.path.join(save_dir, f"Lambda_sl{seq_len}_{suffix}.npy")
    path_lambda_out = os.path.join(save_dir, f"Lambda_pl{pred_len}_{suffix}.npy")

    np.save(path_in, Q_mat.astype(np.float32))
    np.save(path_out, Q_out_mat.astype(np.float32))
    np.save(path_lambda, Lambda_mat.astype(np.float32))
    np.save(path_lambda_out, Lambda_out_mat.astype(np.float32))

    print(f"Frequency Matrices Saved. Q: {Q_mat.shape}, Lambda: {Lambda_mat.shape}")
    return path_in, path_out, path_lambda, path_lambda_out


def _coeffs_to_array(coeffs, length):

    arr = np.concatenate(coeffs, axis=-1)
    if arr.shape[-1] < length:
   
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, length - arr.shape[-1])]
        return np.pad(arr, pad_width)
    else:
     
        return arr[..., :length]





@njit(parallel=True)
def calculate_crps_numba(predictions, ground_truth):
 
    m = predictions.shape[0]
    rest_shape = predictions.shape[1:]
    num_points = 1
    for s in rest_shape:
        num_points *= s
    

    preds_2d = predictions.reshape(m, num_points)
    gt_flat = ground_truth.reshape(num_points)
    
    crps_values = np.empty(num_points, dtype=np.float64)


    for i in prange(num_points):

        sample = np.sort(preds_2d[:, i])
        gt = gt_flat[i]
        

        acc_term = 0.0
        for j in range(m):
            acc_term += abs(sample[j] - gt)
        acc_term /= m
        

        disp_sum = 0.0
        for j in range(m):
            disp_sum += (2 * (j + 1) - 1 - m) * sample[j]
        disp_term = disp_sum / (m * m)
        
        crps_values[i] = acc_term - disp_term

    return np.mean(crps_values)
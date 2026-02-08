from re import X
import torch
import torch.nn.functional as F
import lightning as L
import torch.nn as nn
import numpy as np
import math


class GraphOrthoDiff(L.LightningModule):   
    def __init__(   
        self,   
        backbone: torch.nn.Module, 
        adj_matrix: torch.Tensor, 
        ortho_matrix: torch.Tensor = None, 
        eigen_values: torch.Tensor = None, 
        whitening_power: float = 0.5,
        lr=1e-4,   
        alpha=1e-5,   
        T: int = 1000,   
        beta_schedule: str = 'linear',  
        max_graph_step = 4, 
        **kwargs  
    ) -> None:   
        super().__init__()   
        self.backbone = backbone  
        self.seq_length = self.backbone.seq_length  
        self.num_nodes = adj_matrix.shape[0]  
        
        self.lr = lr  
        self.alpha = alpha  
        self.loss_fn = F.mse_loss  
        self.T = T  
  
        if ortho_matrix is None:
            
            ortho_matrix = torch.eye(self.seq_length)
        
       
        if ortho_matrix.ndim == 2:
            self.indep_mode = False
            # matrix: [T, T]
            Q = ortho_matrix.float()
            Q_inv = Q.T
        elif ortho_matrix.ndim == 3:
            self.indep_mode = True
            # matrix: [N, T, T]
           
            assert ortho_matrix.shape[0] == self.num_nodes, \
                f"Ortho matrix nodes {ortho_matrix.shape[0]} != Model nodes {self.num_nodes}"
            
            Q = ortho_matrix.float()
           
            Q_inv = Q.transpose(1, 2)
        else:
            raise ValueError(f"Invalid ortho_matrix shape: {ortho_matrix.shape}")

        self.register_buffer("Q_mat", Q) 
        self.register_buffer("Q_inv", Q_inv) 


        if eigen_values is not None:
           
            std_orig = torch.sqrt(eigen_values.float().clamp(min=1e-10))
            
            std_scaled = torch.pow(std_orig, whitening_power) 

            if std_scaled.ndim == 1:
                std_reshaped = std_scaled.view(1, -1, 1)
            elif std_scaled.ndim == 2:
                assert std_scaled.shape[0] == self.num_nodes
                std_reshaped = std_scaled.transpose(0, 1).unsqueeze(0)
            
            self.register_buffer('ortho_std', std_reshaped)
            self.enable_whitening = True
            print(f"Partial Whitening Enabled (power={whitening_power}).")
        else:
            self.register_buffer('ortho_std', torch.tensor(1.0))
            self.enable_whitening = False

        if beta_schedule == 'linear':   
            betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)   
        elif beta_schedule == 'cosine':   
            steps = T + 1  
            s = 0.008  
            x_steps = torch.linspace(0, T, steps)   
            alphas_bar = torch.cos(((x_steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2  
            alphas_bar = alphas_bar / alphas_bar[0]   
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])   
            betas = torch.clip(betas, 0.0001, 0.9999)   
        else:   
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")   
  
        alphas = 1. - betas  
        alphas_bar = torch.cumprod(alphas, dim=0)   
  
        self.register_buffer("betas_ddpm", betas)   
        self.register_buffer("alphas_bar", alphas_bar)   
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))   
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))   


        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(adj_matrix)
        except torch.linalg.LinAlgError:
            print("Eigenvalue decomposition failed.")


        K_powers_smooth = torch.zeros(self.T, self.num_nodes, self.num_nodes, device=adj_matrix.device)
        self.lambda_factor = 1 

      
        identity_matrix = torch.eye(self.num_nodes, device=adj_matrix.device)

        for t in range(self.T):
            continuous_step = (t / (self.T - 1)) * max_graph_step
            
            powered_eigenvalues = eigenvalues.abs() ** continuous_step
            
           
            K_t_original = eigenvectors @ torch.diag(powered_eigenvalues) @ eigenvectors.T
            
          
            K_t = (1 - self.lambda_factor) * identity_matrix + self.lambda_factor * K_t_original
            
         
            K_powers_smooth[t] = K_t


        self.register_buffer("K_graph_powers", K_powers_smooth)
        self.save_hyperparameters(ignore=['adj_matrix', 'backbone', 'ortho_matrix'])   
  
    def configure_optimizers(self):   
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.alpha)   

  
    def _forward_ortho(self, x_in):
        """ 
        Time Domain -> Ortho Domain
        x_in: [Batch, Time, Nodes] ('btn')
        target: [Batch, Basis, Nodes] ('bkn')
        """
        Q_on_device = self.Q_mat.to(x_in.device)
        if not self.indep_mode:
            z = torch.einsum('kt,btn->bkn', Q_on_device, x_in)
        else:
        
            z = torch.einsum('nkt,btn->bkn', Q_on_device, x_in)
        if self.enable_whitening:
            # z: [B, K, N]
            # std: [1, K, N]
            z = z / self.ortho_std # Element-wise broadcasting
        return z

    def _inverse_ortho(self, z_in):
        """ 
        Ortho Domain -> Time Domain
        z_in: [Batch, Basis, Nodes] ('bkn')
        """

        if self.enable_whitening:
            # z_in: Latent [B, K, N]
            z_rectified = z_in * self.ortho_std 
        else:
            z_rectified = z_in
        Q_inv_on_device = self.Q_inv.to(z_in.device)
        if not self.indep_mode:
           
            x = torch.einsum('tk,bkn->btn', Q_inv_on_device, z_rectified)
        else:
           
            x = torch.einsum('ntk,bkn->btn', Q_inv_on_device, z_rectified)
        return x
    
    @torch.no_grad()   
    def degrade(self, z_clean: torch.Tensor, t: torch.Tensor):   
        K_power_t_batch = self.K_graph_powers[t]   
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t].view(-1, 1, 1)   
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)   
        z_gcn = torch.einsum('bln,bnm->blm', z_clean, K_power_t_batch)   
        z_filtered = z_gcn * sqrt_alpha_bar_t  
        noise = torch.randn_like(z_filtered)   
        z_noisy = z_filtered + sqrt_one_minus_alpha_bar_t * noise  
        return z_noisy  
  
    def reverse(self, z_t, z_hat, t, prev_t):   
        if t[0] == 0: return z_hat  
        K_t_eff = self.sqrt_alphas_bar[t].view(-1, 1, 1) * self.K_graph_powers[t]   
        prev_t_indices = prev_t.clamp(min=0)   
        K_prev_eff = self.sqrt_alphas_bar[prev_t_indices].view(-1, 1, 1) * self.K_graph_powers[prev_t_indices]   
        beta_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)   
        beta_prev = self.sqrt_one_minus_alphas_bar[prev_t_indices].view(-1, 1, 1)   
        eta_t = self.sigmas[t].view(-1, 1, 1)   
        Kt_z_hat = torch.einsum('bln,bnm->blm', z_hat, K_t_eff)   
        Kprev_z_hat = torch.einsum('bln,bnm->blm', z_hat, K_prev_eff)   
        coeff = torch.sqrt(torch.clamp(beta_prev**2 - eta_t**2, min=0)) / beta_t  
        z_prev = Kprev_z_hat + coeff * (z_t - Kt_z_hat)   
        if (eta_t > 0).any():   
            noise = torch.randn_like(z_t)   
            z_prev += eta_t * noise  
        return z_prev  

    def _get_loss(self, x_norm, condition):   
        batch_size = x_norm.shape[0]   
        cond = condition.get("c", None)   
        t = torch.randint(0, self.T, (batch_size,), device=x_norm.device)   
        z_0 = self._forward_ortho(x_norm)
        z_noisy = self.degrade(z_0, t)   
        z_hat = self.backbone(z_noisy, t, condition=cond, train=True)   
        # x_hat = self._inverse_ortho(z_hat)
        loss = self.loss_fn(z_hat, z_0) 
        # loss = self.loss_fn(x_hat, x_norm)  
        return loss  
  
    def training_step(self, batch, batch_idx):   
        x = batch.pop("x")   
        x_norm = x 
        loss = self._get_loss(x_norm, batch)   
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)   
        return loss   
  
    def validation_step(self, batch, batch_idx):   
        x = batch.pop("x")   
        x_norm = x 
        loss = self._get_loss(x_norm, batch)   
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)   
        return loss  

    def config_sampling(self, n_sample: int = 1, w_cond: float = 1.0, sigmas: torch.Tensor = None, sample_steps=None, **kwargs):   
        self.n_sample = n_sample  
        self.w_cond = w_cond  
        self.sample_Ts = list(range(self.T)) if sample_steps is None else sample_steps  
        self.sample_Ts.sort(reverse=True)   
        effective_betas = self.sqrt_one_minus_alphas_bar  
        sigmas = torch.zeros_like(effective_betas) if sigmas is None else sigmas  
        self.register_buffer("sigmas", sigmas)   
        self._sample_ready = True  
  
    def _init_noise(self, data_shape, device):   
        first_step = self.sample_Ts[0]   
        batch_size = data_shape[0]   
        t = torch.full((batch_size,), first_step, device=device, dtype=torch.long)   
        noise_std = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)   
        z_T = torch.randn(data_shape, device=device) * noise_std  
        return z_T  
  
    def _sample_loop(self, z_curr: torch.Tensor, c: torch.Tensor):   
        for i in range(len(self.sample_Ts)):   
            t_int = self.sample_Ts[i]   
            prev_t_int = self.sample_Ts[i + 1] if i < len(self.sample_Ts) - 1 else 0  
            t_tensor = torch.full((z_curr.shape[0],), t_int, device=z_curr.device, dtype=torch.long)   
            prev_t_tensor = torch.full((z_curr.shape[0],), prev_t_int, device=z_curr.device, dtype=torch.long)   
            
            z_concat = torch.cat([z_curr, z_curr], dim=0)   
            t_concat = torch.cat([t_tensor, t_tensor], dim=0)   
            c_null = torch.zeros_like(c)
            c_concat = torch.cat([c, c_null], dim=0)
            z_hat_concat = self.backbone(z_concat, t_concat, condition=c_concat, train=False)   
            cond_z_hat, uncond_z_hat = torch.chunk(z_hat_concat, 2, dim=0)   
            z_hat = self.w_cond * cond_z_hat + (1 - self.w_cond) * uncond_z_hat  
            z_curr = self.reverse(z_t=z_curr, z_hat=z_hat, t=t_tensor, prev_t=prev_t_tensor)   
        return z_curr  
  
    def predict_step(self, batch, batch_idx):   
        assert hasattr(self, '_sample_ready') and self._sample_ready
        x_real = batch.pop("x")   
        cond = batch.get("c", None)   
        
        all_sample_x = []   
        for _ in range(self.n_sample):   
            z_start = self._init_noise(x_real.shape, x_real.device)   
            z_denoised = self._sample_loop(z_start, cond)   
            x_denoised_norm = self._inverse_ortho(z_denoised)
            out_x = x_denoised_norm 
            all_sample_x.append(out_x)   
        return torch.stack(all_sample_x, dim=0)







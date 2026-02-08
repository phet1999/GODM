import torch
import lightning as L
import pandas as pd
import numpy as np
import argparse
import warnings
import os
import json

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from backbone.ours_model import OLinearAdvancedDenoisingNetwork
from models.gotm import GraphOrthoDiff
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils.utils import *

warnings.filterwarnings('ignore')

def run_experiment(args):
    setup_seed(args.seed)
    print(f"--- Starting Experiment: Dataset={args.dataset}, Seed={args.seed} ---")

    data_path = os.path.join(args.data_dir, args.dataset, 'data.csv')
    adj_path = os.path.join(args.data_dir, args.dataset, 'adj.npy')
    
    df = pd.read_csv(data_path, index_col=0)
    adj = np.load(adj_path)
    train_set, val_set, test_set, scaler = split_and_scale_data(df, 0.8, 0.9)

    NUM_NODES = adj.shape[0]
    
    SEQ_LENGTH = args.seq_len
    COND_LENGTH = args.cond_len
    BATCH_SIZE = args.batch_size
    DIFFUSION_STEPS = args.diffusion_steps
    q_chan_indep = True

    matrix_save_dir = f'./matrix_files/{args.dataset}_{SEQ_LENGTH}/'
    q_path, q_out_path, lambda_path, lambda_out_path = generate_ortho_matrices_lambda(
        np.array(train_set),
        seq_len=COND_LENGTH,
        pred_len=SEQ_LENGTH,
        q_chan_indep=q_chan_indep,
        save_dir=matrix_save_dir
    )
    q_mat = np.load(q_path)
    q_out_mat = np.load(q_out_path)
    lambda_out = torch.Tensor(np.load(lambda_out_path))

    goal = df.columns.to_list()
    train_c, train_x = generate_data_fast(train_set, goal, look_back=COND_LENGTH, look_ahead=SEQ_LENGTH)
    val_c, val_x = generate_data_fast(val_set, goal, look_back=COND_LENGTH, look_ahead=SEQ_LENGTH)
    test_c, test_x = generate_data_fast(test_set, goal, look_back=COND_LENGTH, look_ahead=SEQ_LENGTH)
    
    _, norm_adj_matrix = normalize_adjacency_matrix_efficient(adj)
    
    dm = SpatioTemporalDataModule(train_x, train_c, val_x, val_c, test_x, test_c, BATCH_SIZE)

    
    backbone_config = {
        "enc_in": NUM_NODES, "seq_len": SEQ_LENGTH, "cond_len": COND_LENGTH, "d_model": args.d_model,
        "q_mat": q_mat, "q_out_mat": q_out_mat, "q_chan_indep": True, "p_cond": args.p_cond,
    }
    ortho_tensor = torch.from_numpy(q_out_mat).float()

    my_backbone = OLinearAdvancedDenoisingNetwork(**backbone_config)

    if ortho_tensor.dim() == 3:
        ortho_tensor = ortho_tensor.transpose(1, 2)
    else:
        ortho_tensor = ortho_tensor.transpose(0, 1)

    model_params = {
        "backbone": my_backbone,
        "adj_matrix": norm_adj_matrix,
        "T": DIFFUSION_STEPS,
        "ortho_matrix": ortho_tensor,
        "eigen_values": lambda_out,
    }
    
  
    model_params['max_graph_step'] = args.max_graph_step
    model_params['whitening_power'] = 0.5

    model = GraphOrthoDiff(**model_params)

    logger = TensorBoardLogger("lightning_logs", name=f"{args.dataset}_seed{args.seed}")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best-model-{args.dataset}-seed{args.seed}-pred_length{SEQ_LENGTH}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)
    print("Training finished.")

    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")


    loaded_model = GraphOrthoDiff.load_from_checkpoint(
        best_model_path,
        **model_params 
    )
    
    loaded_model.config_sampling(
        n_sample=args.n_sample,
        w_cond=1.0,
        condition="fcst",
        sample_steps=list(range(DIFFUSION_STEPS - 1, -1, -50))
    )

    predictions_list = trainer.predict(model=loaded_model, datamodule=dm)
    print("Prediction finished.")
    
    qpred = torch.cat(predictions_list, dim=1).numpy()
    pred = np.mean(qpred,0)
    true = np.array(test_x)

    mse = MSE(pred.reshape(-1),true.reshape(-1))
    mae = MAE(pred.reshape(-1),true.reshape(-1))
    crps = calculate_crps_numba(qpred,true)

    print("\n--- Final Metrics ---")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"CRPS: {crps:.4f}")
    print("---------------------\n")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_filename = f"{args.dataset}_pred_length{SEQ_LENGTH}_seed{args.seed}_metrics.json"
    metrics_path = os.path.join(args.output_dir, metrics_filename)
    
    results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'metrics': {
            'MAE': mae,
            'MSE': mse,
            'CRPS': crps
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Metrics saved to {metrics_path}")
    print(f"--- Experiment Finished: Dataset={args.dataset}, Seed={args.seed} ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Orthogonal Diffusion Model Training")

    parser.add_argument('--dataset', type=str, required=True, choices=['PEMS08', 'PEMS07', 'PEMS04', 'PEMS03'], help='Dataset name')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results/', help='Directory to save prediction results')
    
    parser.add_argument('--seq_len', type=int, default=24, help='Length of the prediction sequence')
    parser.add_argument('--cond_len', type=int, default=96, help='Length of the condition sequence')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--p_cond', type=float, default=0.2, help='Conditional dropout probability')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--max_graph_step', type=int, default=4, help='Max graph convolution steps')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    parser.add_argument('--n_sample', type=int, default=50, help='Number of samples to generate for prediction')

    args = parser.parse_args()
    
    run_experiment(args)
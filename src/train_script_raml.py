# %%
import sys
sys.path.append("../modules")

# %%
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from custom_helpers.config import *
from custom_helpers.data_loading import chunk_df, build_loader, construct_fixed_region
from custom_helpers.sample_generation import  generate_samples, load_all_samples
from custom_helpers.scoring import load_scoring_model, construct_mutated_dataset, score_mutated_dataset

from ProteinMPNN.protein_mpnn_utils import tied_featurize, ProteinMPNN as ProteinMPNNVal
from ProteinMPNN.training.model_utils import loss_smoothed, loss_nll, get_std_opt, ProteinMPNN as ProteinMPNNTrain

# %%
class MyArgs(object):
  def __init__(self):
    self.path_for_training_data = "../../data/ppb_affinity"
    self.path_for_outputs = "./test"
    self.previous_checkpoint = ""
    self.num_epochs = 16
    self.save_model_every_n_epochs = 5
    self.reload_data_every_n_epochs = 4
    self.num_examples_per_epoch = 200
    self.total_seq_length = 10000
    self.train_batch_size = 64
    self.val_batch_size = 5
    self.max_protein_length = 2000
    self.hidden_dim = 128
    self.num_encoder_layers = 3
    self.num_decoder_layers = 3
    self.num_neighbors = 48
    self.dropout = 0.1
    self.backbone_noise = 0.02
    self.rescut = 3.5
    self.debug = False
    self.gradient_norm = -1.0 #no norm
    self.mixed_precision= True 
    self.val_batch_copies = 5
    self.sample_temp = 1
    self.lambda_reward = 0.25

main_args = MyArgs()

scoring_args = {
    "checkpoint_path": "/home/ghandill/masterthesis_lilit_ghandilyan/modules/PPB-Affinity-HF/checkpoints/checkpoint.pt",
    "device": torch.device("cuda:3" if (torch.cuda.is_available()) else "cpu"),
}

# %%
split_type = "easy_split"

data_path_base = Path.cwd().parent / "data"
affinity_df = pd.read_csv(data_path_base / 'processed_data.csv', index_col=0)

train_set_df = affinity_df[affinity_df[split_type] == "train"]
val_set_df = affinity_df[affinity_df[split_type] == "val"]

train_chunks = list(chunk_df(train_set_df, main_args.train_batch_size))

# %%
train_model_device = torch.device("cuda:6" if (torch.cuda.is_available()) else "cpu")
val_model_device = torch.device("cuda:7" if (torch.cuda.is_available()) else "cpu")

train_model = ProteinMPNNTrain(node_features=main_args.hidden_dim, 
                    edge_features=main_args.hidden_dim, 
                    hidden_dim=main_args.hidden_dim, 
                    num_encoder_layers=main_args.num_encoder_layers, 
                    num_decoder_layers=main_args.num_encoder_layers, 
                    k_neighbors=48, 
                    dropout=main_args.dropout, 
                    augment_eps=0.02,
                    num_letters=21)
train_model.to(train_model_device)
checkpoint_path = "../modules/ProteinMPNN/vanilla_model_weights/v_48_002.pt"
checkpoint = torch.load(checkpoint_path, map_location=train_model_device)

train_model.load_state_dict(checkpoint['model_state_dict'])

val_model = ProteinMPNNVal(node_features=main_args.hidden_dim, 
                    edge_features=main_args.hidden_dim, 
                    hidden_dim=main_args.hidden_dim, 
                    num_encoder_layers=main_args.num_encoder_layers, 
                    num_decoder_layers=main_args.num_encoder_layers, 
                    k_neighbors=48, 
                    dropout=main_args.dropout, 
                    augment_eps=0.02,
                    num_letters=21)
val_model.to(val_model_device)

for param in val_model.parameters():
    param.requires_grad = False
    
val_model.load_state_dict(train_model.state_dict())

# %%
import wandb
from datetime import datetime

wandb_log = True

if wandb_log:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    wandb.init(
        project="thesis",
        config=main_args.__dict__,
        group="small_batch",
        name=f"time_split_raml_final3_{current_time}",
    )


# %%
import pickle


# def precompute_fixed_positions(df):
#     fixed_positions_dict = {}
#     # Wrap the iterator with tqdm
#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing fixed positions"):
#         row_id = row["complex_id"]
#         fixed_positions_dict[row_id] = construct_fixed_region(row)
#     return fixed_positions_dict

# # Run once
# fixed_positions_cache = precompute_fixed_positions(affinity_df)

# # Save
# with open("fixed_positions_cache.pkl", "wb") as f:
#     pickle.dump(fixed_positions_cache, f)

# Load
with open("cache/fixed_positions_cache.pkl", "rb") as f:
    fixed_positions_cache = pickle.load(f)


# %%
total_step = 0
scaler = torch.amp.GradScaler()

optimizer = optim.Adam(
    train_model.parameters(),
    lr=5e-5  # good default for fine-tuning transformer-type models
)

def train_epoch(train_model, args, epoch, train_device):
    print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")
    train_model.train()
    train_sum, train_weights = 0., 0.
    train_acc = 0.
    
    raml_sum = 0.0
    raml_count = 0

    for _, chunk in tqdm(enumerate(train_chunks), total=len(train_chunks), desc="Train chunks"):
        loader_train, chain_id_dict_train, fixed_positions_dict, _ = build_loader(chunk, args.total_seq_length, fixed_positions_cache)
        for batch in loader_train:
            X, S, mask, _, chain_M, chain_encoding_all, _, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, _, _, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, _ = tied_featurize(
                batch, train_device, chain_id_dict_train, fixed_positions_dict,
                omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict
            )
            optimizer.zero_grad()
            mask_for_loss = mask * chain_M * chain_M_pos
            reward = torch.tensor([-el["score"] for el in batch], device=train_device)
            
            if args.mixed_precision:
                with torch.amp.autocast(device_type="cuda"):
                    log_probs = train_model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    loss_tokenwise, _ = loss_smoothed(S, log_probs, mask_for_loss)

                    # ---- RAML-style weighting ----
                    # Compute per-sequence average loss (masked mean per sequence)
                    seq_loss = torch.sum(loss_tokenwise * mask_for_loss, dim=-1) / (mask_for_loss.sum(dim=-1) + 1e-8)

                    # Compute reward weights
                    weights = torch.exp(args.lambda_reward * reward)  # λR weighting
                    weights = weights / (weights.mean() + 1e-8)       # normalize for stability

                    # Weighted average loss
                    loss_raml = (weights * seq_loss).mean()
                    # -------------------------------
                    
                scaler.scale(loss_raml).backward()
                    
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                log_probs = train_model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all)
                loss_tokenwise, _ = loss_smoothed(S, log_probs, mask_for_loss)

                # ---- RAML-style weighting ----
                seq_loss = torch.sum(loss_tokenwise * mask_for_loss, dim=-1) / (mask_for_loss.sum(dim=-1) + 1e-8)
                weights = torch.exp(args.lambda_reward * reward)
                weights = weights / (weights.mean() + 1e-8)
                loss_raml = (weights * seq_loss).mean()
                # -------------------------------

                loss_raml.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.gradient_norm)

                optimizer.step()
                
            raml_sum += loss_raml.item()
            raml_count += 1

            # still compute these metrics exactly as before
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            train_sum     += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc     += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            avg_raml_loss = raml_sum / max(raml_count, 1)
    return train_model, avg_raml_loss


# %%
def score_subset(scoring_model, subset_df):
    subset_chunks = list(chunk_df(subset_df, main_args.val_batch_size // main_args.val_batch_copies))
    
    all_samples = []

    for _, chunk in tqdm(enumerate(subset_chunks), total=len(subset_chunks), desc="Scoring a subset"):
        samples, chunk_loss = generate_samples(val_model,
                                chunk,
                                device=val_model_device,
                                fixed_positions_cache=fixed_positions_cache,
                                args=main_args)
        
        all_samples.extend(samples)
    mutated_df = construct_mutated_dataset(subset_df, all_samples)
    final_results = score_mutated_dataset(mutated_df, scoring_model, scoring_args)
    return final_results.score.mean(), chunk_loss

# %%
train_loss = None
scoring_model = load_scoring_model(scoring_args)
train_subset = train_set_df.sample(32)
val_subset = val_set_df.sample(32)

for epoch in range(main_args.num_epochs):
    train_score, train_val_loss = score_subset(scoring_model, train_subset)
    val_score, val_val_loss = score_subset(scoring_model, val_subset)
    
    wandb.log({"Train Loss": train_loss, 
               "Train Score": train_score,
               "Vaidation Score": val_score,
               "Validation on train loss": train_val_loss,
               "Epoch": epoch,
               "Validation loss": val_val_loss})
    
    train_model, train_loss = train_epoch(train_model, main_args, epoch, train_model_device)
    val_model.load_state_dict(train_model.state_dict())
    
    torch.save(val_model.state_dict(), f"./model_weights/raml_final3/model_weights_{epoch}.pth")

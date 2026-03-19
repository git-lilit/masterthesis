import gc
import os
import pickle
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime

from custom_helpers.config import *
from custom_helpers.data_loading import construct_dataset
from ProteinMPNN.training.model_utils import loss_smoothed
from ProteinMPNN.protein_mpnn_utils import tied_featurize, _scores, _S_to_seq


def setup_chunk_dir(base_dir="sample_chunks"):
    """Create a timestamped subfolder once per run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")
    return run_dir


def dump_samples(all_sample_list, chunk_counter, chunk_dir):
    """Dump a list of samples to a numbered pickle file in a given directory."""
    chunk_path = os.path.join(chunk_dir, f"samples_{chunk_counter:04d}.pkl")

    with open(chunk_path, "wb") as f:
        pickle.dump(all_sample_list, f)

    print(f"Dumped {len(all_sample_list)} samples to {chunk_path}")

    all_sample_list.clear()
    gc.collect()

    return chunk_counter + 1  # increment chunk counter


def reconstruct_seq(S_tensor, chain_M, masked_chain_length_list, masked_list):
    """Reconstructs the AA sequence string from tensor and chain info."""
    seq = _S_to_seq(S_tensor, chain_M)
    start = 0
    end = 0
    list_of_AAs = []
    for mask_l in masked_chain_length_list:
        end += mask_l
        list_of_AAs.append(seq[start:end])
        start = end

    seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
    l0 = 0
    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
        l0 += mc_length
        seq = seq[:l0] + '/' + seq[l0:]
        l0 += 1
    return seq

def generate_samples(
    val_model,
    val_chunk,
    device,
    fixed_positions_cache,
    args,
    dump_path=None,          # if set, stream results to this jsonl file instead of keeping in memory
    empty_cache_every=20     # how often (per-protein) to call empty_cache / gc.collect
):
    """
    If dump_path is provided, writes one JSON line per sample to the file and returns None.
    Otherwise returns all_sample_list (beware memory).
    """
    val_model.eval()

    # prefer inference_mode if available (faster & lower mem)
    inference_ctx = getattr(torch, "inference_mode", None)
    ctx = inference_ctx if inference_ctx is not None else torch.no_grad

    if dump_path:
        out_f = open(dump_path, "a", encoding="utf-8")
    else:
        all_sample_list = []

    with ctx():
        loss_sum = 0.0
        dataset_valid_chunk, chain_id_dict_chunk, invalid_complexes_valid_chunk, fixed_positions_dict = construct_dataset(val_chunk, fixed_positions_cache)

        for p_ix, protein in enumerate(dataset_valid_chunk):
            # prepare batch clones (these live on CPU until tied_featurize moves them)
            batch_clones = [protein.copy() for _ in range(args.val_batch_copies)]

            (
                X, S, mask, _, chain_M, chain_encoding_all, _, visible_list_list,
                masked_list_list, masked_chain_length_list_list, chain_M_pos,
                omit_AA_mask, residue_idx, _, _, pssm_coef, pssm_bias,
                pssm_log_odds_all, bias_by_res_all, _
            ) = tied_featurize(
                batch_clones,
                device, chain_id_dict_chunk, fixed_positions_dict,
                omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict,
                validation=True
            )

            # remove unused tensor
            # randn_1 = torch.randn(chain_M.shape, device=X.device)  # removed (unused)

            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()

            mask_for_loss = mask * chain_M * chain_M_pos

            # Generate samples
            randn_2 = torch.randn(chain_M.shape, device=X.device)
            sample_dict = val_model.sample(
                X, randn_2, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=args.sample_temp, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag),
                pssm_log_odds_mask=pssm_log_odds_mask,
                pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all
            )

            S_sample = sample_dict["S"]

            # Compute model score for sampled sequences
            log_probs = val_model(
                X, S_sample, mask, chain_M * chain_M_pos, residue_idx,
                chain_encoding_all, randn_2,
                use_input_decoding_order=True,
                decoding_order=sample_dict["decoding_order"]
            )
            
            # Calculating the loss for plotting 
            reward = torch.tensor(-protein["score"])
            loss_tokenwise, _ = loss_smoothed(S, log_probs, mask_for_loss)

            seq_loss = torch.sum(loss_tokenwise * mask_for_loss, dim=-1) / (mask_for_loss.sum(dim=-1) + 1e-8)

            # Compute reward weights
            weights = torch.exp(args.lambda_reward * reward)  # λR weighting
            # weights = weights / (weights.mean() + 1e-8)       # normalize for stability

            # Weighted average loss
            loss_raml = (weights * seq_loss).mean()
            # -------------------------------
            
            loss_sum += loss_raml

            scores = _scores(S_sample, log_probs, mask_for_loss).cpu().numpy()

            # Move necessary tensors to CPU (detach first to be safe)
            S_sample_cpu = S_sample.detach().cpu().clone()
            S_cpu = S.detach().cpu().clone()
            chain_M_cpu = chain_M.detach().cpu().clone()

            # iterate over batch copies
            for b_ix in range(args.val_batch_copies):
                masked_chain_length_list = masked_chain_length_list_list[b_ix]
                masked_list = masked_list_list[b_ix]

                sample_seq = reconstruct_seq(
                    S_sample_cpu[b_ix], chain_M_cpu[b_ix],
                    masked_chain_length_list_list[b_ix], masked_list_list[b_ix]
                )

                if b_ix == 0:
                    native_seq = reconstruct_seq(S_cpu[b_ix], chain_M_cpu[b_ix], masked_chain_length_list, masked_list)
                    print_masked_chains = [masked_list_list[0][i] for i in np.argsort(masked_list_list[0])]
                    print_visible_chains = [visible_list_list[0][i] for i in np.argsort(visible_list_list[0])]

                score_cpu = float(scores[b_ix])
                residue_idx_cpu = protein["resns"].detach().cpu().numpy().copy() if torch.is_tensor(protein["resns"]) else np.array(protein["resns"]).copy()

                result_dict = {
                    "protein_id": protein["name"],
                    "sample_n": int(b_ix),
                    "sample_seq": str(sample_seq),
                    "native_seq": str(native_seq),
                    "sample_score": np.format_float_positional(np.float32(score_cpu), unique=False, precision=4),
                    "designed_chains": [str(x) for x in print_masked_chains],
                    "residue_idx": residue_idx_cpu.tolist() if isinstance(residue_idx_cpu, np.ndarray) else residue_idx_cpu
                }

                if dump_path:
                    out_f.write(json.dumps(result_dict) + "\n")
                else:
                    all_sample_list.append(result_dict)

            # --- Clean up big tensors explicitly ---
            # delete tensors that live on GPU so their memory can be freed sooner
            to_del = [
                X, S, mask, chain_M, chain_encoding_all,
                pssm_log_odds_all, pssm_log_odds_mask, pssm_coef, pssm_bias,
                bias_by_res_all, mask_for_loss, randn_2, sample_dict, log_probs, scores,
                S_sample, S_sample_cpu, S_cpu, chain_M_cpu
            ]
            for _v in to_del:
                try:
                    del _v
                except Exception:
                    pass

            # Also delete any names still referencing GPU tensors (residue_idx_cpu etc are on CPU)
            gc.collect()

            # periodically release CUDA cache (helps fragmentation)
            if (p_ix + 1) % empty_cache_every == 0:
                torch.cuda.empty_cache()
                # optional sync:
                # torch.cuda.synchronize()

        # end for proteins

        # final cleanup for chunk
        del dataset_valid_chunk, chain_id_dict_chunk, invalid_complexes_valid_chunk, fixed_positions_dict
        gc.collect()
        torch.cuda.empty_cache()

    if dump_path:
        out_f.close()
        return None
    else:
        if len(all_sample_list) != 0:
            n_generated = len(all_sample_list)
            loss_sum = loss_sum / n_generated
        return all_sample_list, loss_sum

# def generate_samples(val_model, val_chunks, device, fixed_positions_cache, args):
#     val_model.eval()
#     chunk_dir = setup_chunk_dir()
#     chunk_counter = 1

#     with torch.inference_mode():
#         all_sample_list = []
#         for _, chunk in tqdm(enumerate(val_chunks), total=len(val_chunks), desc="Val chunks"):
#             dataset_valid_chunk, chain_id_dict_chunk, invalid_complexes_valid_chunk, fixed_positions_dict = construct_dataset(chunk, fixed_positions_cache)
#             for _, protein in enumerate(dataset_valid_chunk):
#                 batch_clones = [protein.copy() for _ in range(args.val_batch_copies)]

#                 (
#                     X, S, mask, _, chain_M, chain_encoding_all, _, visible_list_list,
#                     masked_list_list, masked_chain_length_list_list, chain_M_pos,
#                     omit_AA_mask, residue_idx, _, _, pssm_coef, pssm_bias,
#                     pssm_log_odds_all, bias_by_res_all, _
#                 ) = tied_featurize(
#                     batch_clones, 
#                     device, chain_id_dict_chunk, fixed_positions_dict,
#                     omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict,
#                     validation=True
#                 )

#                 pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()

#                 randn_1 = torch.randn(chain_M.shape, device=X.device)
#                 mask_for_loss = mask * chain_M * chain_M_pos

#                 # Generate samples
#                 randn_2 = torch.randn(chain_M.shape, device=X.device)
#                 sample_dict = val_model.sample(
#                     X, randn_2, S, chain_M, chain_encoding_all, residue_idx,
#                     mask=mask, temperature=args.sample_temp, omit_AAs_np=omit_AAs_np,
#                     bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos,
#                     omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias,
#                     pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag),
#                     pssm_log_odds_mask=pssm_log_odds_mask,
#                     pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all
#                 )

#                 S_sample = sample_dict["S"]

#                 # Compute model score for sampled sequences
#                 log_probs = val_model(
#                     X, S_sample, mask, chain_M * chain_M_pos, residue_idx,
#                     chain_encoding_all, randn_2,
#                     use_input_decoding_order=True,
#                     decoding_order=sample_dict["decoding_order"]
#                 )

#                 scores = _scores(S_sample, log_probs, mask_for_loss).cpu().numpy()
                
#                 # Move to CPU before reconstruction
#                 S_sample_cpu = S_sample.detach().cpu()
#                 S_cpu = S.detach().cpu()
#                 chain_M_cpu = chain_M.detach().cpu()
                    
#                 # Iterate over batch copies
#                 for b_ix in range(args.val_batch_copies):
#                     masked_chain_length_list = masked_chain_length_list_list[b_ix]
#                     masked_list = masked_list_list[b_ix]

#                     sample_seq = reconstruct_seq(S_sample_cpu[b_ix], chain_M_cpu[b_ix],
#                                          masked_chain_length_list_list[b_ix], masked_list_list[b_ix])

#                     if b_ix == 0:
#                         native_seq = reconstruct_seq(S_cpu[b_ix], chain_M_cpu[b_ix], masked_chain_length_list, masked_list)
#                         print_masked_chains = [masked_list_list[0][i] for i in np.argsort(masked_list_list[0])]
#                         print_visible_chains = [visible_list_list[0][i] for i in np.argsort(visible_list_list[0])]

#                     score_cpu = float(scores[b_ix])  # ensures it's a plain Python float

#                     # Make sure residue_idx is on CPU (if it's a tensor)
#                     residue_idx_cpu = protein["resns"].detach().cpu().numpy() if torch.is_tensor(protein["resns"]) else protein["resns"]

#                     result_dict = {
#                         "protein_id": protein["name"],
#                         "sample_n": int(b_ix),
#                         "sample_seq": str(sample_seq),
#                         "native_seq": str(native_seq),
#                         "sample_score": np.format_float_positional(np.float32(score_cpu), unique=False, precision=4),
#                         "designed_chains": [str(x) for x in print_masked_chains],
#                         "residue_idx": residue_idx_cpu
#                     }

#                     all_sample_list.append(result_dict)
                    
#                 if len(all_sample_list) > 100:
#                     chunk_counter = dump_samples(all_sample_list, chunk_counter, chunk_dir)

#                 del X, S, mask, chain_M, chain_encoding_all
#                 del pssm_log_odds_all, pssm_log_odds_mask, pssm_coef, pssm_bias
#                 del bias_by_res_all, mask_for_loss, randn_2, sample_dict, log_probs, scores
#             del dataset_valid_chunk, chain_id_dict_chunk, invalid_complexes_valid_chunk
            
#             torch.cuda.empty_cache()
#             gc.collect()
#         dump_samples(all_sample_list, chunk_counter, chunk_dir)

#     return chunk_dir


def load_all_samples(chunk_dir):
    """Load and combine all sample chunks from a given directory."""
    all_samples = []
    
    # Find all pickle files in the directory
    chunk_files = sorted(glob(os.path.join(chunk_dir, "samples_*.pkl")))
    
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        return all_samples

    for file_path in chunk_files:
        with open(file_path, "rb") as f:
            samples = pickle.load(f)
            all_samples.extend(samples)
            print(f"Loaded {len(samples)} samples from {os.path.basename(file_path)}")

    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples

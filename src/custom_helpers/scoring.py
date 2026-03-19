import sys
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from custom_helpers.data_loading import parse_mutation_string

sys.path.append("../modules/PPBAffinity")
from inference import InferDataset_Struc
from BaselineModel.utils.data import PaddingCollate_struc
from BaselineModel.utils.train import CrossValidation, recursive_to
from BaselineModel.models.dg_model import DG_Network


def construct_mutation_strings(sample):
    mutation_strings = []
    residue_indices = sample["residue_idx"]
    
    native_seq_chain_map = dict(zip(sample["designed_chains"], sample["native_seq"].split("/")))
    sample_seq_chain_map = dict(zip(sample["designed_chains"], sample["sample_seq"].split("/")))

    for chain in sample["designed_chains"]:
        for idx, residue_idx in enumerate(residue_indices[chain]):
            old_letter = native_seq_chain_map[chain][idx]
            new_letter = sample_seq_chain_map[chain][idx]
            if old_letter == new_letter:
                continue
            mutation_string = f'{chain}_{old_letter}{residue_idx + 1}{new_letter}'
            mutation_strings.append(mutation_string)
            
    return mutation_strings


def load_scoring_model(args):
    ckpt = torch.load(args["checkpoint_path"], map_location=args["device"], weights_only=False)
    config = ckpt['config']

    model = CrossValidation(
        config=config,
        model_factory=DG_Network,
        num_cvfolds=config['train']['num_cvfolds'],
    )
    model.load_state_dict(ckpt['model'])
    model.to(args["device"])
    
    return model


def construct_mutated_dataset(val_set_df, all_samples):
    rows = []

    for sample in all_samples:
        match = val_set_df[val_set_df.complex_id == sample["protein_id"]]
        match['mutstr'] = ",".join(construct_mutation_strings(sample))
        if not match.empty:
            rows.append(match.iloc[0])

    return pd.DataFrame(rows)

def score_mutated_dataset(mutated_df, scoring_model, args):
    dataset = InferDataset_Struc(mutated_df)
    
    valid_entries = []
    for i in range(len(dataset)):
        try:
            _ = dataset[i]
            valid_entries.append(dataset.entries[i])
        except:
            pass

    dataset.entries = valid_entries

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=PaddingCollate_struc(),
        num_workers=8
    )

    result = []
    for batch in tqdm(loader):
        
        batch = recursive_to(batch, args["device"])
        for fold in range(scoring_model.num_cvfolds):
            model, _, _ = scoring_model.get(fold)
            model.eval()
            all_dG_true = []
            all_dg_pred = []
            with torch.no_grad():
                pred_dG = model.infer(batch)
            for pdbname, dG_true, dG_pred, mutstr, original_complex_id in zip(batch['ID'], 
                                                        batch['dG'].cpu().tolist(),
                                                        pred_dG.cpu().tolist(),
                                                        batch["mutstr"],
                                                        batch["complex_id"]):
                result.append({
                    'ID': pdbname,
                    'dG_true': dG_true,
                    'dG_pred': dG_pred,
                    'mutstr': mutstr,
                    'complex_id': original_complex_id
                })
                
    result = pd.DataFrame(result)

    final_result = []
    for ID, df in result.groupby('ID'):
        dG_pred = df['dG_pred'].mean()
        dG_true = df['dG_true'].mean()
        mutstr = df['mutstr'].iloc[0]
        complex_id = df['complex_id'].iloc[0]
        final_result.append({
            'ID': ID,
            'dG_true': dG_true,
            'dG_pred': dG_pred,
            'mutstr': mutstr,
            'complex_id': complex_id
        })
            
    final_result_df = pd.DataFrame(final_result)
    final_result_df['score'] =  final_result_df['dG_true'] - final_result_df['dG_pred']

    return final_result_df
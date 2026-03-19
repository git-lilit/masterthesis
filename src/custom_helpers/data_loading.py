import re
import torch
import pandas as pd
from pathlib import Path
from custom_helpers.config import *

from ProteinMPNN.training.utils import StructureLoader
from ProteinMPNN.protein_mpnn_utils import parse_PDB, StructureDatasetPDB

from Bio.PDB.PDBParser import PDBParser
from PPBAffinity.BaselineModel.utils.protein.parsers import parse_biopython_structure
from PPBAffinity.BaselineModel.utils.transforms import _index_select_data
from PPBAffinity.BaselineModel.utils.transforms import Compose, SelectAtom, SelectedRegionFixedSizePatch


project_root = Path.cwd().parent # adjust as needed
data_path_base = project_root / "data"

def parse_mutation_string(mutstr):
    # Updated regex: allows optional negative positions
    pattern = r"(?P<chain>[A-Za-z0-9])?_?(?P<orig>[A-Z])(?P<pos>-?\d+)(?P<new>[A-Z])"

    match = re.match(pattern, mutstr)
    if not match:
        raise ValueError(f"Invalid mutation format: {mutstr}")

    chain = match.group("chain")
    original_letter = match.group("orig")
    position = int(match.group("pos"))
    new_letter = match.group("new")
    
    return chain, original_letter, position, new_letter


def get_pdb_path(row):
    if row["source"] == "SAbDab":
            pdb_file_name = f"{row['pdb']}.pdb"
    elif row["source"] == "PDBbind v2020":
        pdb_file_name = f"{row['pdb']}.ent.pdb"
    else:
        pdb_file_name = f"{row['pdb'].upper()}.pdb"
        
    pdb_path = data_path_base / "ppb_affinity" / "pdb" / row["source"] / pdb_file_name
    
    return pdb_path


def construct_fixed_region(row):
    transform = Compose([
                SelectAtom('full'),
                SelectedRegionFixedSizePatch('itf_flag', 128)
            ])

    parser = PDBParser(QUIET=True)
    model = parser.get_structure(None, row["pdb_path"])[0]
    
    # Fix missing occupancies before parsing
    for atom in model.get_atoms():
        if atom.occupancy is None:
            atom.set_occupancy(1.0)
    
    data, seq_map = parse_biopython_structure(model) 

    group_id = []
    for ch in data['chain_id']:
        if ch in row['ligand'].split(", "):  # 1 is ligand
            group_id.append(1)
        elif ch in row['receptor'].split(", "):  # 2 is receptor
            group_id.append(2)
        else:
            group_id.append(0)
    data['group_id'] = torch.LongTensor(group_id)

    idx_keep = torch.where(data['group_id'] != 0)[0]
    data = _index_select_data(data, idx_keep)

    idx_ligand = torch.where(data['group_id'] == 1)[0]
    idx_receptor = torch.where(data['group_id'] == 2)[0]
    dist_pair = torch.cdist(data['pos_heavyatom'][idx_ligand, 1, :],
                            data['pos_heavyatom'][idx_receptor, 1, :])  # 1 is CA atom

    idx_ligand_itf, idx_receptor_itf = torch.where(dist_pair < 10.0)
    idx_ligand_itf = idx_ligand[torch.unique(idx_ligand_itf)]
    idx_receptor_itf = idx_receptor[torch.unique(idx_receptor_itf)]
    idx_itf = torch.cat([idx_ligand_itf, idx_receptor_itf])
    data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
    data['itf_flag'][idx_itf] = True
    if data['itf_flag'].sum()==0:
            raise ValueError("No binding site found")
        
    data = transform(data)

    binding_pocket_region = {}
    for idx, (chain, resseq) in enumerate(zip(data['chain_id'], data['resseq'].tolist())):
        if data['itf_flag'][idx].item():
            binding_pocket_region.setdefault(chain, []).append(resseq)

    fixed_positions = {}

    for chain in model:
        chain_id = chain.id
        if chain_id not in binding_pocket_region.keys(): continue
        for idx, residue in enumerate(chain):
            if residue.id[0] != " ": continue
            if not residue.id[1] in binding_pocket_region[chain_id]:
                fixed_positions.setdefault(chain_id, []).append(idx)
                
    return fixed_positions


def construct_dataset(subset_df, fixed_positions_cache):
    pdb_dict_list = []
    invalid_complexes = []
    chain_id_dict = {}
    fixed_region_dict = {}

    for idx, row in subset_df.iterrows():
        row_id = row["complex_id"]
        pdb_path = get_pdb_path(row)

        # check if pdb file exists
        if not pdb_path.exists():
            print(f"Skipping {row_id}: {pdb_path} not found")
            continue

        fixed_positions = fixed_positions_cache[row_id]
        fixed_region_dict[row_id] = fixed_positions

        ligand_chain_list = row["ligand"].split(", ")
        all_chain_list = row["ligand"].split(", ") + row["receptor"].split(", ")
        
        designed_chain_list = [chain for chain in ligand_chain_list if chain in fixed_positions.keys()]
        fixed_chain_list = [chain for chain in all_chain_list if chain not in designed_chain_list]
        
        chain_id_dict[row_id] = (designed_chain_list, fixed_chain_list)
   
        mutations = row["mutstr"]
        mut_dict = {}

        if not pd.isna(mutations):
            for mutstr in mutations.split(", "):
                chain, original_letter, position, new_letter = parse_mutation_string(mutstr)
                mut_dict.setdefault(chain, []).append((original_letter, position, new_letter))
        
        parsed_pdb = parse_PDB(pdb_path, input_chain_list=all_chain_list, unique_identifier=row_id, mut_dict=mut_dict)[0]
        parsed_pdb['score'] = row['dG']
        parsed_pdb['pdb'] = row['pdb']

        dict_keys_list = parsed_pdb.keys()
        valid_chains = [key.split('_')[-1] for key in dict_keys_list if key.startswith('seq_chain')]
        all_chains_exist = all(chain_letter in valid_chains for chain_letter in all_chain_list)
        
        if all_chains_exist: 
            pdb_dict_list.extend([parsed_pdb])
        else:
            invalid_complexes.append(row_id)
        
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

    return dataset_valid, chain_id_dict, invalid_complexes, fixed_region_dict

def chunk_df(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def build_loader(df_chunk, total_seq_length, fixed_positions_cache):
    dataset, chain_id_dict, invalid_complexes_chunk, fixed_region_dict = construct_dataset(df_chunk, fixed_positions_cache)
    loader = StructureLoader(dataset, batch_size=total_seq_length)
    return loader, chain_id_dict, fixed_region_dict, invalid_complexes_chunk

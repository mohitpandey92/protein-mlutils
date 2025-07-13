import numpy as np
import re
from Bio.Align import PairwiseAligner


def mutation_over_base_seq(base_seq,new_seq, starting_pos=1):
    
    '''
    Note that position starts with Q=1
    Input:
    base_seq: str, sequence of amino acids
    new_seq: str, sequence of amino acids
    starting_pos: int, should we use starting_pos=0 vs starting_pos=1 naming convention

    '''
    assert len(base_seq)==len(new_seq), "expected same length sequence"

    

    #if len(base_seq)
    base_seq_list=np.array([aa for aa in base_seq])
    new_seq_list=np.array([aa for aa in new_seq])

    aa_pos_diff_arr=np.where(new_seq_list!=base_seq_list)[0]
    mutation_list=[]
    for pos in aa_pos_diff_arr:
        if starting_pos==1:
            mutation_temp=base_seq_list[pos]+str(pos+1)+new_seq_list[pos]
        elif starting_pos==0:
            mutation_temp=base_seq_list[pos]+str(pos)+new_seq_list[pos]
        else:
            raise ValueError("Unexpected Q index")
        mutation_list=mutation_list+[mutation_temp]
    return mutation_list


def mutate_sequence_fn(base_seq, mutation_list, starting_pos=1):
    '''
    Input:
    base_seq: str, base sequence that needs to be mutated
    mutation_list: list, list of mutations that need to be applied
    starting_pos: int, should we use starting_pos=0 vs starting_pos=1 naming convention
    Output:
    mutated_seq: str, protein sequence with mutations
    '''
    
    base_seq_list=[aa for aa in base_seq]
    mutated_seq_list=base_seq_list.copy()
    
    pos_arr= np.zeros(len(mutation_list), dtype=int)
    mut_arr=np.zeros(len(mutation_list), dtype=str)
    for index, mut in enumerate(mutation_list):
        if starting_pos==1:
            pos=int(mut[1:-1]) -1
        elif starting_pos==0:
            pos=int(mut[1:-1])
        else:
            raise ValueError("Unexpected starting_pos index")
        pos_arr[index]=pos
        mut_arr[index]=mut[-1]
    
    for index,pos in enumerate(pos_arr):
        mutated_seq_list[pos]=mut_arr[index]
    return  ''.join(mutated_seq_list)

aa_three_letter={'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe', 'G': 'Gly',
                'H': 'His', 'I': 'Ile', 'K':'Lys', 'L': 'Leu', 'M': 'Met', 'N':'Asp', 'P':'Pro', 'Q': 'Gln', 'R': 'Arg', 
                 'S': 'Ser', 'U': 'Sec', 'V':'Val', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr'}

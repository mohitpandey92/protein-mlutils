import numpy as np
import torch
import torch.nn as nn

def protein_one_hot_encoder(sequence, max_length=None, flatten=True ):
    """
    For a given protein sequence, this function performs one-hot encoding. If given a max_length, it pads the sequence with underscores to that length.
    If the sequence is longer than a given max_length, it will not truncate the sequence.
    Args:
        sequence (str): The protein sequence to encode.
        max_length (int, optional): The maximum length of the sequence. Defaults to None.

    Returns:
        np.ndarray: A one-hot encoded representation of the protein sequence.
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY_'
    assert isinstance(max_length, (int, type(None))), "max_length must be an integer or None"
    
    assert len(amino_acids) == 21, "There should be 21 amino acids in the encoding"

    encoding = {aa: i for i, aa in enumerate(amino_acids)}

    if isinstance(sequence, str):
        if max_length is not None:
            assert len(sequence) <= max_length, "Sequence length exceeds max length"
        assert all(aa in amino_acids for aa in sequence), "Sequence contains invalid amino acids"
        if (max_length is not None) and len(sequence) < max_length:
            sequence += '_' * (max_length - len(sequence))  # Padding with underscores if needed
        else:
            max_length = len(sequence)
        
        # Initialize one-hot encoding matrix

        one_hot = np.zeros((len(amino_acids), max_length), dtype=int)


        for i, aa in enumerate(sequence):
            if aa in encoding:
                one_hot[encoding[aa]][i] = 1
                
        if flatten:
            one_hot = one_hot.flatten(order='F')
        return one_hot
    elif isinstance(sequence, (list, np.ndarray)):
        # If the sequence is a list, we need to process each element
        one_hot_list = []
        for seq in sequence:
            one_hot_list.append(protein_one_hot_encoder(seq, max_length, flatten))
        return np.array(one_hot_list)
    else:
        raise ValueError("Sequence must be a string, list, or numpy array")






def protein_embedding_generator(sequence, max_length=None, embedding_dim=5, flatten=True):
    """
    Generates an embedding for a given protein sequence using PyTorch's nn.Embedding. These embeddings are generated from a random distribution. This is an alternative to one-hot encoding whose output is a large but sparse matrix. This embedding is dense and can be compressed to a given embedding dimension. It is likely to be more efficient for training deep neural networks.
    Args:
        sequence (str or list): The protein sequence to encode.
        max_length (int, optional): The maximum length of the sequence. Defaults to None.

    Returns:
        embedding_arr(np.ndarray): An embedding representation of the protein sequence.
        embedding_generator_weights(np.ndarray): The weights of the embedding layer. This can be used to initialize the nn.embedding layer so that we can use the same embeddings for other sequences during inference time. nn.Embedding(num_embeddings=10,embedding_dim=3, _weight=embedding_generator_weights)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY_'
    amino_acids_dict = {aa: i for i, aa in enumerate(amino_acids)}
    assert isinstance(max_length, (int, type(None))), "max_length must be an integer or None"
    if isinstance(sequence, str):
    
        if (max_length is not None) and len(sequence) < max_length:
                sequence += '_' * (max_length - len(sequence))  # Padding with underscores if needed
        sequence_index_list=[]
        for i, aa in enumerate(sequence):
            sequence_index_list.append(amino_acids_dict[aa])

        embedding_generator=nn.Embedding(len(amino_acids),embedding_dim=embedding_dim)
        output=embedding_generator(torch.LongTensor(sequence_index_list))


    elif isinstance(sequence, (list, np.ndarray)):
        # If the sequence is a list, we need to process each element
        
        sequence_index_big_list=[]
        for seq in sequence:
            if (max_length is not None) and len(seq) < max_length:
                seq += '_' * (max_length - len(seq))  # Padding with underscores if needed
            sequence_index_list=[]
            for i, aa in enumerate(seq):
                sequence_index_list.append(amino_acids_dict[aa])
            sequence_index_big_list.append(sequence_index_list)

        embedding_generator=nn.Embedding(len(amino_acids),embedding_dim=embedding_dim)
        #it's important to generate the
        output=embedding_generator(torch.LongTensor(sequence_index_big_list))
        
    else:
        raise ValueError("Sequence must be a string, list, or numpy array")

    if flatten: #return the output as a flattened 1D array
        output=output.detach().numpy().flatten()
    else:  #return the output as is
        output=output.detach().numpy()

    return output, embedding_generator.weight.detach().numpy()
    
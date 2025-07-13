#consolidate all the amino acid code
hydrophobicity_roster = {
        "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48, 
        "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78,
        "P": 0.12, "Q": -0.85, "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08, 
        "W": 0.81, "Y": 0.26
        }

hydrophilicity_roster = {
        "A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5,  "G": 0.0, 
        "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8, "M": -1.3, "N": 0.2, 
        "P": 0.0, "Q": 0.2, "R": 3.0, "S": 0.3, "T": -0.4, "V": -1.5, 
        "W": -3.4, "Y": -2.3
        }

max_hydrophobicity_roster={'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0,'L': 1,'M': 0,'N': 0, 'P': 0,'Q': 0,'R': 0,'S': 0,'T': 0, 'V': 1,'W': 1, 'Y': 0}

charge_roster= {
        "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, "F":0.0,  "G": 0.0, 
        "H": 0.0, "I": 0.0, "K": 1.0, "L": 0.0, "M": 0.0, "N": 0.0, 
        "P": 0.0, "Q": 0.0, "R": 1.0, "S": 0.0, "T": 0.0, "V": 0.0, 
        "W": 0.0, "Y": 0.0
        }

positive_charge_roster= {
        "A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F":0.0,  "G": 0.0, 
        "H": 0.0, "I": 0.0, "K": 1.0, "L": 0.0, "M": 0.0, "N": 0.0, 
        "P": 0.0, "Q": 0.0, "R": 1.0, "S": 0.0, "T": 0.0, "V": 0.0, 
        "W": 0.0, "Y": 0.0, "-": 0.0
        }

negative_charge_roster= {
        "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, "F":0.0,  "G": 0.0, 
        "H": 0.0, "I": 0.0, "K": 0.0, "L": 0.0, "M": 0.0, "N": 0.0, 
        "P": 0.0, "Q": 0.0, "R": 0.0, "S": 0.0, "T": 0.0, "V": 0.0, 
        "W": 0.0, "Y": 0.0, "-": 0.0,
        }

def total_charge_fn(seq, type='net_charge'):
    '''
    It computes the net charge on physiological pH
    '''
    if type=='net_charge':
        net_charge=0.0
        for aa_i in seq:
            net_charge=net_charge+charge_roster[aa_i]
    
        return net_charge
    
    elif type=='positive_charge':
        pos_charge=0.0
        for aa_i in seq:
            pos_charge=pos_charge+positive_charge_roster[aa_i]
    
        return pos_charge
    
    elif type=='negative_charge':
        neg_charge=0.0
        for aa_i in seq:
            neg_charge=neg_charge+negative_charge_roster[aa_i]
    
        return neg_charge  
   
    elif type=='hydrophobicity':
        hydrophobicity=0.0
        for aa_i in seq:
            hydrophobicity=hydrophobicity+max_hydrophobicity_roster[aa_i]
            
        return hydrophobicity
        
    
    else:
        raise ValueError("Unknown type")

def average_property_fn(seq, property="hydrophobicity"):
    avg_property=0
    if property=="hydrophobicity":
        for aa_i in seq:
            avg_property=avg_property+hydrophobicity_roster[aa_i]

    elif property=="hydrophilicity":
        for aa_i in seq:
            avg_property=avg_property+hydrophilicity_roster[aa_i]

    elif property=="positive_charge":
        for aa_i in seq:
            avg_property=avg_property+positive_charge_roster[aa_i]

    elif property=="negative_charge":
        for aa_i in seq:
            avg_property=avg_property+negative_charge_roster[aa_i]

    elif property=="net_charge":
        for aa_i in seq:
            avg_property=avg_property+charge_roster[aa_i]
    
    else:
        return ValueError("unknown property")
    return avg_property/len(seq)
        

def CasX_domain_boundary_fn(pos):
    '''
    #OBD = 1-59, 499-642
    #HelI = 60-99, 197-337
    #NTSB = 100-196
    #HelII = 338-498
    #RuvC = 643-694, 750-812, 914-978
    #BH = 695-749
    #TSL = 813-913

    '''
    if (1<=pos<=59) or (499<=pos<=642):
        return "OBD" #PAM recognition site
    elif (60<=pos<=99) or (197<=pos<=337):
        return "HelI" #PAM recognition site
    elif (100<=pos<=196):
        return "NTSB" #important for DNA unfolding
    elif (338<=pos<=498):
        return "HelII"
    elif (643<=pos<=694) or (750<=pos<=812) or (914<=pos<=978):
        return "RuvC"
    elif (695<=pos<=749):
        return "BH"
    elif (813<=pos<=913):
        return "TSL"
    else:
        raise ValueError("Unexpected pos")

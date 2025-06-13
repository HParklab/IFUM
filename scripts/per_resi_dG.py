import os
from pymol import cmd
import gzip
import numpy as np

one_letter = {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
    'GLY':'G', 'PRO':'P', 'CYS':'C'
}

def pdg_to_color(pdg, min_pdg, max_pdg):
    # map pdg: low -> red, 0+ -> white
    pdg = np.clip(pdg, min_pdg, max_pdg)
    if pdg <= 0:
        # red to white
        t = (pdg - min_pdg) / (0 - min_pdg) if min_pdg < 0 else 1.0
        r, g, b = 255, int(255*t), int(255*t)
    else:
        # white to blue
        t = pdg / max_pdg if max_pdg > 0 else 0.0
        r = int(255*(1 - t))
        g = int(255*(1 - t))
        b = 255

    color = "0x%02x%02x%02x" % (r, g, b)
    return color

pdbnames = cmd.get_object_list()

for pdbname in pdbnames:
    seqs = []
    pdgs = []

    with open(f"{pdbname}.pdb", 'r') as f:
        for line in f:
            if not "per_resi_dg" in line:
                continue

            sp = line.strip().split()
            seqpos = int(sp[0].split("_")[-1])
            pdg = float(sp[1])

            seqs.append(seqpos)
            pdgs.append(pdg)


    # determine actual data range
    # min_pdg = min(pdgs)
    # max_pdg = max(pdgs)
    min_pdg = -0.12
    max_pdg = 0

    low_pdgs = []
    
    for seqpos in range(1, max(seqs)+1):
        if ( seqpos not in seqs):
            continue
        idx = seqs.index(seqpos)
        pdg = pdgs[idx]

        color = pdg_to_color(pdg, min_pdg, max_pdg)

        if ( pdg < 0 ):
            cmd.color(color, f"resi {seqpos} and {pdbname}")
            # retrieve the CA atom to get its 3-letter resn, map to one-letter, then label
            model = cmd.get_model(f"resi {seqpos} and name CA and {pdbname}")
            if model.atom and pdg < -0.08:
                resn3 = model.atom[0].resn
                resn1 = one_letter.get(resn3, 'X')
                cmd.label(f"resi {seqpos} and name CA and {pdbname}", f"\"{resn1}{seqpos}: {pdg}\"")
                low_pdgs.append(seqpos)
        
        else:
            cmd.color("white", f"resi {seqpos} and {pdbname}")
        
    if low_pdgs:
        cmd.select(f"{pdbname}_low_pdgs", f"resi {'+'.join(map(str, low_pdgs))} and {pdbname}")

cmd.deselect()
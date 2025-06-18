import os
from pymol import cmd
import numpy as np

one_letter = {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
    'GLY':'G', 'PRO':'P', 'CYS':'C'
}

def read_pdb( fname ):
    if os.path.exists(f"{fname}.pdb"):
        return open(f"{fname}.pdb")
    elif os.path.exists(f"{fname}.cif"):
        return open(f"{fname}.cif")
    else:
        return False

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

object_names = cmd.get_object_list()

for name in object_names:
    f = read_pdb(name)
    if not f:
        print(f"Could not find PDB or CIF file for {name}. Skipping.")
        continue

    seqs = []
    pdgs = []

    for line in f:
        if "# per_resi_dg" not in line:
            continue

        sp = line.strip().split()
        seqpos = int(sp[1].split("_")[-1])
        pdg = float(sp[2])

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
            cmd.color(color, f"resi {seqpos} and {name}")
            # retrieve the CA atom to get its 3-letter resn, map to one-letter, then label
            model = cmd.get_model(f"resi {seqpos} and name CA and {name}")
            if model.atom and pdg < -0.08:
                resn3 = model.atom[0].resn
                resn1 = one_letter.get(resn3, 'X')
                cmd.label(f"resi {seqpos} and name CA and {name}", f"\"{resn1}{seqpos}: {pdg}\"")
                low_pdgs.append(seqpos)

        else:
            cmd.color("white", f"resi {seqpos} and {name}")

    if low_pdgs:
        cmd.select(f"{name}_low_pdgs", f"resi {'+'.join(map(str, low_pdgs))} and {name}")

cmd.deselect()
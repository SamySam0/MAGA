import rdkit.Chem as Chem
#from rdkit.Chem.Draw import MolToImage
import matplotlib.pyplot as plt

def graphs_to_mols(graphs):
    '''
    Args:
        A list of torch_geometric Data.
    '''
    mols = []
    for graph in graphs:
        mols.append(graph_to_mol(graph))
    return mols

def graph_to_mol(graph):
    '''
    Args:
        graph (torch_geometric Data):
            THe bond type is assume to be one-hot encoded in edge_attr with 4 being the aromatic bond
    '''
    bond_types = (Chem.BondType.SINGLE,
                  Chem.BondType.DOUBLE,
                  Chem.BondType.TRIPLE,
                  Chem.BondType.AROMATIC)
    mol = Chem.RWMol()
    for node in graph.x:
        if node.sum() > 0:
            node = node.argmax(-1) + 6
            mol.AddAtom(Chem.Atom(int(node)))
    for idx, edge in enumerate(graph.edge_index.T):
        atom1, atom2 = edge
        atom1 = int(atom1.item())
        atom2 = int(atom2.item())
        if atom1 < atom2:
            idx = graph.edge_attr[idx].argmax()
            mol.AddBond(atom1, atom2, bond_types[idx])
    return mol

def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]

def check_validity(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

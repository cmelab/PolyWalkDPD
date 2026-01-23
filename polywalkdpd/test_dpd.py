import logging
from mupt.builders.dpd import DPDRandomWalk, LOGGER as DPD_LOGGER

loghandler = logging.StreamHandler()
loghandler.setLevel(logging.DEBUG)
DPD_LOGGER.addHandler(loghandler)

import numpy as np
R_excl : float = 10.0
bond_length : float = 1.5 # 5.5
angle_max_rad : float = np.pi/4
n_chains : int = 10
dop_min : int = 50 # must be at least 2!
dop_max : int = 150

from mupt.geometry.shapes import Ellipsoid, Sphere
from mupt.geometry.coordinates.reference import CoordAxis, origin
from mupt.geometry.transforms.rigid import rigid_vector_coalignment
from mupt.mupr.primitives import Primitive, PrimitiveHandle
from mupt.interfaces.rdkit import primitive_to_rdkit, suppress_rdkit_logs
from mupt.interfaces.smiles import primitive_from_smiles

import networkx as nx
from mupt.mupr.topology import TopologicalStructure
from mupt.builders.dpd import DPDRandomWalk
from mupt.geometry.coordinates.directions import random_unit_vector

AXIS : CoordAxis = CoordAxis.X
SEMIMINOR_FRACT : float = 0.5 # how long the pair of minor axes should be as a fraction of the major axis length

rep_unit_smiles : dict[str, str] = {
    'head' : f'[H]-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
    'mid'  : f'*-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
    'tail' : f'*-[O:1]c1ccc(cc1)S(=O)(=O)c1ccc(cc1)[O:2]-[H]',
}

lexicon : dict[str, Primitive] = {}
with suppress_rdkit_logs():
    for unit_name, smiles in rep_unit_smiles.items():
        unitprim = primitive_from_smiles(smiles, ensure_explicit_Hs=True, embed_positions=True, label=unit_name)
            
        # force edge atoms to lie along chosen axis, with midpoint at the origin
        head_atom, tail_atom = unitprim.search_hierarchy_by(lambda prim : 'molAtomMapNumber' in prim.metadata, min_count=2)
        head_pos, tail_pos = head_atom.shape.centroid, tail_atom.shape.centroid
        
        major_radius = np.linalg.norm(tail_pos - head_pos) / 2.0
        axis_vec = np.zeros(3, dtype=float)
        axis_vec[AXIS.value] = major_radius
        axis_alignment = rigid_vector_coalignment(
            vector1_start=head_pos,
            vector1_end=tail_pos,
            vector2_start=origin(3),
            vector2_end=axis_vec,
            t1=1/2,
            t2=0.0,
        )
        unitprim.rigidly_transform(axis_alignment)
        lexicon[unit_name] = unitprim
        semiminor = SEMIMINOR_FRACT * major_radius
        radii = np.full(3, semiminor)
        radii[AXIS.value] = major_radius
        unitprim.shape = Ellipsoid(radii) # = Sphere(major_radius)
        rdmol = primitive_to_rdkit(unitprim)

univprim = Primitive(label='universe')
for chain_len in np.random.randint(dop_min, dop_max + 1, size=n_chains):
    molprim = Primitive(label=f'{chain_len}-mer_chain')
    unit_names : list[str] = ['head'] + ['mid']*(chain_len - 2) + ['tail']
    for i, unit_name in enumerate(unit_names)   :
        rep_unit_prim = lexicon[unit_name].copy()
        molprim.attach_child(rep_unit_prim)           
    molprim.set_topology(
        nx.path_graph(
            molprim.children_by_handle.keys(),
            create_using=TopologicalStructure,
        ),
        max_registration_iter=100,
    )
    mol_handle = univprim.attach_child(molprim)
    univprim.expand(mol_handle)
builder = DPDRandomWalk(output_name="init")
for handle,placement in builder.generate_placements(univprim):
    #print(placement)
    pass

#univprim.visualize_topology()
#print(univprim.hierarchy_summary(to_depth=2))



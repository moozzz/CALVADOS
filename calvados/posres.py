"""Position restraints and constant-force pulling on the centres of mass of selected beads.

Added by patch_calvados_position_restraints.py. Enabled in config.yaml with
position_restraints = True; fposition_restraints is a YAML file with a list of entries,
each a position restraint (k) or a constant pulling force (force and direction):

- selection: (chain Y and resid 682 to 691) or (chain Z and resid 442 to 451)
  k: 100.0         # restraint: force constant (kJ/mol/nm^2)
  per: unit YZ     # selection (default) | chain | bead | unit <chain-ID patterns>
  r0: 0.0          # restraint: flat-bottom radius (nm), no force within r0 of the target
  dims: xyz        # restraint: restrained directions, e.g. xy
  anchor: chain I J K L M N O P Q R   # optional, see below
  component: name  # optional if only one component has an input structure

- selection: (chain Y and resid 682 to 691) or (chain Z and resid 442 to 451)
  per: unit YZ
  force: 20.0          # pulling: constant force (kJ/mol/nm; 1 kJ/mol/nm = 1.66 pN)
  direction: [0, 0, -1]  # pulling: direction of the force (normalised)
  anchor: chain I J K L M N O P Q R   # optional: the anchor feels the opposite force

selection, anchor: VMD-style (chain, resid, resname, name, index, residue, ranges
  "a to b", and/or/not, parentheses) or MDAnalysis syntax, evaluated on the input
  structure of the component (<pdb_folder>/<component>.pdb or .cif), so chain IDs and
  residue numbers are those of that file; a selected atom selects its bead (residue).
per: one restraint or force on the centre of mass (COM) of all selected beads (selection),
  of the selected beads of each chain (chain), of each complex (unit: consecutive chains
  matching one of the chain-ID patterns, e.g. "unit YZ" or "unit AB IJKLMNOPQR YZ";
  chains matching no pattern are units of their own), or of each bead (bead). Groups
  never span molecules (nmol > 1: every copy is treated separately).
anchor: restraint without anchor: each COM is restrained to its position in the start
  structure; with anchor: the COM position relative to the COM of the anchor beads (one
  anchor COM per entry and molecule) is restrained to its start value, so the restraint
  follows translations of the anchor (e.g. a drifting microtubule-ring assembly).
  Pulling with anchor: the anchor COM feels the opposite force (no net external force).

Restraint: E = 0.5 k max(0, |d| - r0)^2, d = minimum-image displacement of the COM from
its target (absolute or relative to the anchor COM) in the restrained directions.
Pulling: E = -force n.d with the unit vector n of direction, i.e. a constant force
force*n on the COM (distributed over its beads in proportion to their masses) and -force*n
on the anchor COM. d is not minimum-imaged, so the force is exact for any pulled distance;
the pulling energy jumps by force*n.L if OpenMM moves the pulled molecule by a box vector
L (GPU platforms; forces and dynamics are not affected). COMs are weighted by the bead
masses. Targets come from the start structure built from the input (top.pdb), also when
the run continues from a checkpoint.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import openmm
import yaml
from MDAnalysis import Universe
from MDAnalysis.exceptions import NoDataError
from openmm import app, unit
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from .sim import Sim

RESTRAINT = (
    "0.5*k*max(0, d - r0)^2;"
    "d = pointdistance(x1, y1, z1, select(mx, {X}, x1), select(my, {Y}, y1), select(mz, {Z}, z1))"
)
PULL = "-f*(nx*(x1 - ({X})) + ny*(y1 - ({Y})) + nz*(z1 - ({Z})))"
PARAMETERS = {
    "restraint": ["k", "r0", "x0", "y0", "z0", "mx", "my", "mz"],
    "pull": ["f", "x0", "y0", "z0", "nx", "ny", "nz"],
}
KJ_PER_NM_IN_PN = 1.66054  # 1 kJ/mol/nm in pN
VMD_KEYWORDS = {"chain": "chainID", "residue": "resindex", "segname": "segid"}


class PositionRestraintInput(BaseModel):
    """One entry of the position-restraint file: a restraint (k) or a pulling force."""

    model_config = ConfigDict(extra="forbid")

    selection: str
    k: PositiveFloat | None = None
    per: str = "selection"
    r0: NonNegativeFloat = 0.0
    dims: str = "xyz"
    force: float | None = None
    direction: tuple[float, float, float] | None = None
    anchor: str | None = None
    component: str | None = None

    @field_validator("per")
    @classmethod
    def check_per(cls, per: str) -> str:
        words = per.split()
        if not words or words[0] not in ("selection", "chain", "bead", "unit"):
            raise ValueError("per must be selection, chain, bead or unit <chain-ID patterns>")
        if (words[0] == "unit") != (len(words) > 1):
            raise ValueError("per: unit needs chain-ID patterns (e.g. unit YZ), the others none")
        return " ".join(words)

    @field_validator("dims")
    @classmethod
    def check_dims(cls, dims: str) -> str:
        if not dims or set(dims) - set("xyz") or len(set(dims)) != len(dims):
            raise ValueError("dims must be a combination of x, y and z, e.g. xyz or xy")
        return dims

    @model_validator(mode="after")
    def check_kind(self) -> Self:
        if self.force is None:
            if self.k is None:
                raise ValueError("a restraint needs k (or give force and direction for pulling)")
            if self.direction is not None:
                raise ValueError("direction needs force (pulling)")
        else:
            if self.force == 0:
                raise ValueError("force must not be zero")
            if self.direction is None or not np.linalg.norm(self.direction) > 0:
                raise ValueError("pulling needs a non-zero direction, e.g. [0, 0, 1]")
            given = {"k", "r0", "dims"} & self.model_fields_set
            if given:
                raise ValueError(f"{', '.join(sorted(given))}: only for restraints, not for pulling")
        return self


def read_input(fname: Any) -> list[PositionRestraintInput]:
    """Read and validate the position-restraint file."""
    with open(fname) as f:
        entries = yaml.safe_load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{fname}: expected a non-empty list of position restraints")
    return [PositionRestraintInput.model_validate(e) for e in entries]


def to_mdanalysis(selection: str) -> str:
    """Translate VMD selection keywords to MDAnalysis (MDAnalysis syntax passes unchanged)."""
    sel = re.sub(r"(-?\d+)\s+to\s+(-?\d+)", r"\1:\2", selection)
    return re.sub(r"\b(chain|residue|segname)\b", lambda m: VMD_KEYWORDS[m.group(1)], sel)


class Structure:
    """Input structure of a component: bead selection and chain/unit membership."""

    def __init__(self, fname: str, nbeads: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.u = Universe(app.PDBxFile(fname)) if fname.lower().endswith(".cif") else Universe(fname)
        self.fname = fname
        if len(self.u.residues) != nbeads:
            raise ValueError(
                f"{fname}: {len(self.u.residues)} residues, but the component has {nbeads} "
                "beads; position restraints need one bead per residue"
            )
        self.chain_of_bead = self.u.residues.segindices  # chains = segments, as in CALVADOS
        try:
            ids = self.u.atoms.chainIDs
        except NoDataError:
            ids = self.u.atoms.segids
        self.chain_ids = [ids[seg.atoms[0].index] for seg in self.u.segments]

    def beads(self, selection: str) -> np.ndarray:
        """0-based beads (residues) with at least one selected atom."""
        try:
            atoms = self.u.select_atoms(to_mdanalysis(selection))
        except Exception as e:
            raise ValueError(f"invalid selection {selection!r}: {e}") from e
        if len(atoms) == 0:
            raise ValueError(f"selection {selection!r} matches no atoms in {self.fname}")
        return np.unique(atoms.resindices)

    def unit_of_chain(self, patterns: list[str]) -> np.ndarray:
        """Units: consecutive chains whose IDs match one of the patterns (else single chains)."""
        if any(len(c) != 1 for c in self.chain_ids):
            raise ValueError(f"{self.fname}: per unit needs one-character chain IDs")
        ids = "".join(self.chain_ids)
        patterns = sorted(patterns, key=len, reverse=True)
        unit, n, k = np.zeros(len(ids), dtype=int), 0, 0
        while k < len(ids):
            size = next((len(p) for p in patterns if ids.startswith(p, k)), 1)
            unit[k:k + size] = n
            n, k = n + 1, k + size
        return unit

    def groups(self, beads: np.ndarray, per: str) -> list[np.ndarray]:
        """Split the selected beads into the restrained groups."""
        mode, *patterns = per.split()
        if mode == "selection":
            return [beads]
        if mode == "bead":
            return [beads[i:i + 1] for i in range(len(beads))]
        key = self.chain_of_bead[beads]
        if mode == "unit":
            key = self.unit_of_chain(patterns)[key]
        return [beads[key == v] for v in np.unique(key)]

    def describe(self, group: np.ndarray) -> str:
        """Chains of a group, e.g. Y#329+Z#330 (1-based chain numbers in the file)."""
        return "+".join(f"{self.chain_ids[c]}#{c + 1}" for c in np.unique(self.chain_of_bead[group]))


def ranges(beads: np.ndarray) -> str:
    """1-based bead numbers as ranges, e.g. 99245-99254,100075-100084."""
    b = np.asarray(beads) + 1
    breaks = np.flatnonzero(np.diff(b) != 1)
    starts, ends = np.r_[b[0], b[breaks + 1]], np.r_[b[breaks], b[-1]]
    return ",".join(f"{s}-{e}" if e > s else f"{s}" for s, e in zip(starts, ends))


def make_force(kind: str, anchored: bool) -> openmm.CustomCentroidBondForce:
    """CustomCentroidBondForce for restraints or pulling; group 2 is the anchor."""
    target = ("x2 + x0", "y2 + y0", "z2 + z0") if anchored else ("x0", "y0", "z0")
    expression = (RESTRAINT if kind == "restraint" else PULL).format(X=target[0], Y=target[1], Z=target[2])
    force = openmm.CustomCentroidBondForce(2 if anchored else 1, expression)
    for p in PARAMETERS[kind]:
        force.addPerBondParameter(p)
    # restraints: minimum-image distances; pulling: plain coordinate differences
    force.setUsesPeriodicBoundaryConditions(kind == "restraint")
    force.setName(("PositionRestraints" if kind == "restraint" else "Pulling")
                  + ("Anchored" if anchored else ""))
    return force


def summary(forces: list[openmm.Force]) -> str:
    """Number of restraints and pulled groups, as printed by Sim.add_forces_to_system."""
    npull = sum(f.getNumBonds() for f in forces if f.getName().startswith("Pulling"))
    text = f"Number of position restraints: {sum(f.getNumBonds() for f in forces) - npull}"
    return text + (f"\nNumber of pulled groups: {npull}" if npull else "")


def build(sim: Sim) -> list[openmm.Force]:
    """Create the restraint and pulling forces from the start structure of a built system."""
    entries = read_input(sim.config.fposition_restraints)
    pos = np.asarray(sim.pos, dtype=float)
    mass = np.array([sim.system.getParticleMass(i).value_in_unit(unit.dalton)
                     for i in range(sim.system.getNumParticles())])

    start, total = {}, 0  # first bead of each component, in the order of build_system
    for comp in sim.components:
        start[comp.name] = total
        total += comp.params.nmol * comp.nbeads
    # components whose beads come from an input structure
    with_structure = [c for c in sim.components if c.params.restraint and c.params.nmol > 0]
    structures: dict[str, Structure] = {}

    forces: dict[tuple[str, bool], openmm.CustomCentroidBondForce] = {}
    records = []
    for n, entry in enumerate(entries, 1):
        if entry.component is None:
            if len(with_structure) != 1:
                raise ValueError(f"position restraint {n}: give the component (components "
                                 "with an input structure: "
                                 f"{', '.join(c.name for c in with_structure) or 'none'})")
            comp = with_structure[0]
        else:
            comp = next((c for c in with_structure if c.name == entry.component), None)
            if comp is None:
                raise ValueError(f"position restraint {n}: no component {entry.component!r} "
                                 "with an input structure")
        if comp.name not in structures:
            structures[comp.name] = Structure(
                comp.get_input_structure_file(comp.params.pdb_folder, comp.name), comp.nbeads)
        struct = structures[comp.name]
        groups = struct.groups(struct.beads(entry.selection), entry.per)
        anchor = None if entry.anchor is None else struct.beads(entry.anchor)

        kind = "restraint" if entry.force is None else "pull"
        if kind == "restraint":
            params = [entry.k, entry.r0]
            flags = [float(d in entry.dims) for d in "xyz"]
            columns = f"restraint {entry.k:g} {entry.r0:g} {entry.dims} - - - -"
        else:
            direction = np.asarray(entry.direction, dtype=float)
            direction /= np.linalg.norm(direction)
            params, flags = [entry.force], direction.tolist()
            columns = f"pull - - - {entry.force:g} " + " ".join(f"{x:.6f}" for x in direction)
        key = (kind, anchor is not None)
        if key not in forces:
            forces[key] = make_force(*key)
        force = forces[key]

        for copy in range(comp.params.nmol):
            offset = start[comp.name] + copy * comp.nbeads
            if anchor is not None:
                a = anchor + offset
                anchor_group = force.addGroup(a.tolist(), mass[a].tolist())
                anchor_com = mass[a] @ pos[a] / mass[a].sum()
            for g in groups:
                b = g + offset
                com = mass[b] @ pos[b] / mass[b].sum()
                target = com if anchor is None else com - anchor_com
                group = force.addGroup(b.tolist(), mass[b].tolist())
                bond = [group] if anchor is None else [group, anchor_group]
                force.addBond(bond, [*params, *target, *flags])
                records.append(
                    f"{len(records) + 1} {n} {comp.name} {copy + 1} {struct.describe(g)} {columns} "
                    f"{target[0]:.4f} {target[1]:.4f} {target[2]:.4f} {len(b)} {ranges(b)} "
                    f"{'-' if anchor is None else ranges(anchor + offset)}\n")

        what = {"selection": "the selection", "chain": "each chain", "bead": "each bead"}.get(
            entry.per, "each " + entry.per)
        sizes = sorted({len(g) for g in groups})
        nbeads = "-".join(map(str, sizes[::max(1, len(sizes) - 1)]))
        if kind == "restraint":
            print(f"Position restraint {n}: {len(groups) * comp.params.nmol} COM restraint(s) on "
                  f"{what} ({nbeads} beads), k = {entry.k:g} kJ/mol/nm^2, r0 = {entry.r0:g} nm, "
                  f"dims {entry.dims}"
                  + ("" if anchor is None else f", relative to the COM of {len(anchor)} anchor beads"))
        else:
            print(f"Pulling {n}: constant force {entry.force:g} kJ/mol/nm "
                  f"({entry.force * KJ_PER_NM_IN_PN:.3g} pN) along "
                  f"({', '.join(f'{x:.3g}' for x in direction)}) on the COM of {what} "
                  f"({len(groups) * comp.params.nmol} group(s) of {nbeads} beads)"
                  + ("" if anchor is None else
                     f", opposite force on the COM of {len(anchor)} anchor beads"))

    with open(f"{sim.path}/posres_{sim.config.sysname}.txt", "w") as f:
        f.write("# restraint: E = 0.5 k max(0, |d| - r0)^2, d = minimum-image displacement of "
                "COM(beads) from its target in the restrained dims\n"
                "# pull: constant force f along n on COM(beads) [and -f n on COM(anchor)], "
                "E = -f n.d, d = displacement of COM(beads) from its target\n"
                "# target t: start COM, or start COM - COM(anchor) with an anchor; beads 1-based\n"
                "# id entry component copy chains type k[kJ/mol/nm^2] r0[nm] dims f[kJ/mol/nm] "
                "nx ny nz tx[nm] ty[nm] tz[nm] nbeads beads anchor\n")
        f.writelines(records)
    return list(forces.values())

"""Position restraints, constant-force and constant-velocity pulling on the centres of mass
of selected beads.

Added by patch_calvados_position_restraints.py (patch version 3). Enabled in config.yaml with
position_restraints = True; fposition_restraints is a YAML file with a list of entries,
each a position restraint (k), a constant pulling force (force and direction) or
constant-velocity pulling (velocity, k and direction):

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

- selection: (chain Y and resid 682 to 691) or (chain Z and resid 442 to 451)
  per: unit YZ
  velocity: 10.0       # constant velocity: speed of the spring's target (nm/ns)
  k: 100.0             # constant velocity: spring constant along the direction (kJ/mol/nm^2)
  direction: [0, 0, -1]
  update_steps: 100    # optional: steps between moves of the target (default 100)
  anchor: chain I J K L M N O P Q R   # optional: target moves relative to the anchor COM

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
L (GPU platforms; forces and dynamics are not affected).
Constant-velocity pulling: E = 0.5 k (n.dt)^2 with dt the minimum-image displacement of
the COM from its moving target, origin + velocity*t*n (relative to the anchor COM), so only
the component along n is restrained (the reaction acts on the anchor COM). The pull starts
with the production run (the spring is off during the energy minimisation and any
equilibration): the origin is the COM at that moment and t the time since then, so the
spring starts relaxed. dt is the lag behind the target; it, and for a direction that is
not along a box axis also the sideways drift of the COM, must stay below half the box
length minus 1 nm. PullReporter moves the targets every update_steps steps (the smallest
value of all entries) to the middle of the next interval, adds up the work done by moving
them (the jumps of the spring energy, exact for this stepwise protocol) and writes
pull_<sysname>.txt every logfreq steps. The start time, the origins and the work are kept
in restart.chk, so a continuation carries on exactly. A new pull starts, and the previous
log is kept as backup_pull_<sysname>_<date>.txt, in a new run, with restart = 'pdb'/'cif',
or when the velocity entries (groups, k, velocity, direction) changed.
COMs are weighted by the bead masses. Targets of restraints and constant forces come from
the start structure built from the input (top.pdb), also when the run continues from a
checkpoint (topol 'random': the targets of the first run, from posres_<sysname>.txt).
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from datetime import datetime
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
    PositiveInt,
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
# spring along n to the moving target t = X + v*pull_time*n; n.(COM - t) from the minimum
# images of COM - t -/+ n: |a + n|^2 - |a - n|^2 = 4 n.a
VELOCITY = (
    "pull_on*0.5*k*p^2;"
    "p = (pointdistance(x1, y1, z1, tx - nx, ty - ny, tz - nz)^2"
    " - pointdistance(x1, y1, z1, tx + nx, ty + ny, tz + nz)^2)/4;"
    "tx = {X} + nx*v*pull_time; ty = {Y} + ny*v*pull_time; tz = {Z} + nz*v*pull_time"
)
PARAMETERS = {
    "restraint": ["k", "r0", "x0", "y0", "z0", "mx", "my", "mz"],
    "pull": ["f", "x0", "y0", "z0", "nx", "ny", "nz"],
    "velocity": ["k", "v", "x0", "y0", "z0", "nx", "ny", "nz"],
}
PULL_UPDATE_STEPS = 100  # default steps between moves of the constant-velocity targets
PULL_FORCE_GROUP = 30    # force group of the constant-velocity springs (work bookkeeping)
POSRES_VERSION = 3  # 1: restraints, constant force; 2: + constant velocity; 3: + pull state in restart.chk
KJ_PER_NM_IN_PN = 1.66054  # 1 kJ/mol/nm in pN
VMD_KEYWORDS = {"chain": "chainID", "residue": "resindex", "segname": "segid"}


class PositionRestraintInput(BaseModel):
    """One entry of the position-restraint file: a restraint (k), a constant pulling force
    (force) or constant-velocity pulling (velocity)."""

    model_config = ConfigDict(extra="forbid")

    selection: str
    k: PositiveFloat | None = None
    per: str = "selection"
    r0: NonNegativeFloat = 0.0
    dims: str = "xyz"
    force: float | None = None
    velocity: float | None = None
    direction: tuple[float, float, float] | None = None
    update_steps: PositiveInt | None = None
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

    @property
    def kind(self) -> str:
        return "pull" if self.force is not None else "velocity" if self.velocity is not None else "restraint"

    @model_validator(mode="after")
    def check_kind(self) -> Self:
        if self.force is not None and self.velocity is not None:
            raise ValueError("give force (constant force) or velocity (constant velocity), not both")
        if self.update_steps is not None and self.kind != "velocity":
            raise ValueError("update_steps: only for constant-velocity pulling")
        if self.kind == "restraint":
            if self.k is None:
                raise ValueError("a restraint needs k (or give force or velocity and direction for pulling)")
            if self.direction is not None:
                raise ValueError("direction needs force or velocity (pulling)")
            return self
        if (self.force if self.kind == "pull" else self.velocity) == 0:
            raise ValueError(f"{'force' if self.kind == 'pull' else 'velocity'} must not be zero")
        if self.direction is None or not np.linalg.norm(self.direction) > 0:
            raise ValueError("pulling needs a non-zero direction, e.g. [0, 0, 1]")
        if self.kind == "velocity" and self.k is None:
            raise ValueError("constant-velocity pulling needs the spring constant k")
        given = ({"k", "r0", "dims"} if self.kind == "pull" else {"r0", "dims"}) & self.model_fields_set
        if given:
            raise ValueError(f"{', '.join(sorted(given))}: not for this kind of pulling")
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
            if fname.lower().endswith(".cif"):
                cif = app.PDBxFile(fname)
                self.u = Universe(cif)
                # MDAnalysis numbers the residues of an OpenMM topology 1..N: use the file's numbers
                self.u.residues.resids = np.array([int(r.id) for r in cif.topology.residues()])
            else:
                self.u = Universe(fname)
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
    template = {"restraint": RESTRAINT, "pull": PULL, "velocity": VELOCITY}[kind]
    expression = template.format(X=target[0], Y=target[1], Z=target[2])
    force = openmm.CustomCentroidBondForce(2 if anchored else 1, expression)
    for p in PARAMETERS[kind]:
        force.addPerBondParameter(p)
    if kind == "velocity":
        force.addGlobalParameter("pull_time", 0.0)  # ps since the start of the pull, set by PullReporter
        force.addGlobalParameter("pull_on", 0.0)  # switched on by PullReporter (off during minimisation)
        force.setForceGroup(PULL_FORCE_GROUP)
    # restraints and constant velocity: minimum image; constant force: plain coordinates
    force.setUsesPeriodicBoundaryConditions(kind != "pull")
    force.setName({"restraint": "PositionRestraints", "pull": "Pulling", "velocity": "PullingVelocity"}[kind]
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
    records, bonds = [], []  # bonds: (force, index, groups, params, flags, absolute target)
    sim.posres_pulls = []  # pulled groups, for PullReporter
    sim.posres_mass = mass
    sim.posres_update_steps = min((e.update_steps or PULL_UPDATE_STEPS for e in entries
                                   if e.kind == "velocity"), default=PULL_UPDATE_STEPS)
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

        kind = entry.kind
        if kind == "restraint":
            params = [entry.k, entry.r0]
            flags = [float(d in entry.dims) for d in "xyz"]
            columns = f"restraint {entry.k:g} {entry.r0:g} {entry.dims} - - - - -"
        else:
            direction = np.asarray(entry.direction, dtype=float)
            direction /= np.linalg.norm(direction)
            flags = direction.tolist()
            if kind == "pull":
                params = [entry.force]
                columns = f"pull - - - {entry.force:g} - "
            else:
                params = [entry.k, entry.velocity / 1000.0]  # nm/ns -> nm/ps
                columns = f"velocity {entry.k:g} - - - {entry.velocity:g} "
            columns += " ".join(f"{x:.6f}" for x in direction)
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
                index = force.addBond(bond, [*params, *target, *flags])
                bonds.append((force, index, bond, params, flags, anchor is None))
                if kind != "restraint":
                    sim.posres_pulls.append(dict(
                        kind=kind, label=struct.describe(g), beads=b, target=target, n=direction,
                        anchor=None if anchor is None else anchor + offset,
                        k=entry.k, f=entry.force, v=None if entry.velocity is None else entry.velocity / 1000.0,
                        force=force, bond=index, groups=bond))
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
        elif kind == "velocity":
            print(f"Pulling {n}: constant velocity {entry.velocity:g} nm/ns with a spring of "
                  f"k = {entry.k:g} kJ/mol/nm^2 along ({', '.join(f'{x:.3g}' for x in direction)}) "
                  f"on the COM of {what} ({len(groups) * comp.params.nmol} group(s) of {nbeads} beads), "
                  f"target moved every {sim.posres_update_steps} steps"
                  + ("" if anchor is None else
                     f", target relative to the COM of {len(anchor)} anchor beads"))
        else:
            print(f"Pulling {n}: constant force {entry.force:g} kJ/mol/nm "
                  f"({entry.force * KJ_PER_NM_IN_PN:.3g} pN) along "
                  f"({', '.join(f'{x:.3g}' for x in direction)}) on the COM of {what} "
                  f"({len(groups) * comp.params.nmol} group(s) of {nbeads} beads)"
                  + ("" if anchor is None else
                     f", opposite force on the COM of {len(anchor)} anchor beads"))

    fname = f"{sim.path}/posres_{sim.config.sysname}.txt"
    keep = False
    if sim.config.restart == "checkpoint" and sim.restart_path.is_file() and os.path.isfile(fname):
        # continuation: keep the absolute targets of the first run if the molecules were
        # placed differently this time (e.g. topol 'random'); same targets otherwise
        saved = [line.split() for line in open(fname) if line.strip() and not line.startswith("#")]
        if len(saved) == len(records) and all(s[18] == r.split()[18] for s, r in zip(saved, records)):
            for s, r, (force, index, groups, params, flags, absolute) in zip(saved, records, bonds):
                old, new = np.array(s[14:17], dtype=float), np.array(r.split()[14:17], dtype=float)
                if absolute and s[5] != "velocity" and np.abs(old - new).max() > 1e-3:
                    force.setBondParameters(index, groups, [*params, *old, *flags])
                    keep = True
            if keep:
                print(f"Position restraints: targets of the first run taken from {fname}")
    vel = [p for p in sim.posres_pulls if p["kind"] == "velocity"]
    if vel:
        # pull state saved in restart.chk: identity of the pulling setup, start time, work
        # and the origin of every target (set by PullReporter)
        key = repr([(ranges(p["beads"]), None if p["anchor"] is None else ranges(p["anchor"]),
                     p["k"], p["v"], [round(float(x), 6) for x in p["n"]]) for p in vel])
        sim.posres_pull_id = float(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
        host = vel[0]["force"]
        for name in ("pull_id", "pull_t0", "pull_work"):
            host.addGlobalParameter(name, 0.0)
        for i, p in enumerate(vel, 1):
            p["index"] = i
            for c in "xyz":
                host.addGlobalParameter(f"pull_o{i}_{c}", 0.0)
    if keep:
        return list(forces.values())
    with open(fname, "w") as f:
        f.write("# restraint: E = 0.5 k max(0, |d| - r0)^2, d = minimum-image displacement of "
                "COM(beads) from its target in the restrained dims\n"
                "# pull: constant force f along n on COM(beads) [and -f n on COM(anchor)], "
                "E = -f n.d, d = displacement of COM(beads) from its target\n"
                "# velocity: spring k along n to the target moving at v, E = 0.5 k (n.(COM(beads) - "
                "origin - v time n))^2 (minimum image), origin = COM at the start of the pull\n"
                "# target t: start COM, or start COM - COM(anchor) with an anchor; beads 1-based\n"
                "# id entry component copy chains type k[kJ/mol/nm^2] r0[nm] dims f[kJ/mol/nm] "
                "v[nm/ns] nx ny nz tx[nm] ty[nm] tz[nm] nbeads beads anchor\n")
        f.writelines(records)
    return list(forces.values())


class PullReporter:
    """OpenMM reporter for the pulled groups (attached by Sim.simulate).

    Constant velocity: the pull starts when the reporter is attached (after the energy
    minimisation) with every target at the current COM of its group (relative to the
    anchor COM). Every update_steps steps the targets move to the middle of the next
    interval (pull_time = time since the start + update_steps*dt/2), and the work done by
    moving them, the change of the spring energy at fixed positions, is added up. The start
    time, the origins of the targets and the work are context parameters, so restart.chk
    keeps them and a continuation carries on exactly. A new pull (targets at the current
    COMs, work 0) starts when the checkpoint holds no pull with the same groups, k, velocity
    and direction: a new run, restart = 'pdb'/'cif', or changed velocity entries.
    Every logfreq steps, pull_<sysname>.txt gets the time, the pull time, the total work and,
    for every pulled group, its COM displacement along n (relative to the anchor COM;
    constant velocity: since the start of the pull, constant force: from the start
    structure) and the force on it along n.
    """

    def __init__(self, sim: Sim, simulation: app.Simulation, append: bool):
        self.pulls = sim.posres_pulls
        self.mass = sim.posres_mass
        self.vel = [p for p in self.pulls if p["kind"] == "velocity"]
        self.nupdate = sim.posres_update_steps
        self.nlog = sim.config.logfreq
        self.dt = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        context = simulation.context
        t = context.getTime().value_in_unit(unit.picosecond)
        self.t0, self.work, note = 0.0, 0.0, None
        continuing = append
        if self.vel:
            continuing = append and context.getParameter("pull_id") == sim.posres_pull_id
            if continuing:  # the same pull: origins, start time and work from the checkpoint
                for p in self.vel:
                    p["origin"] = np.array([context.getParameter(f"pull_o{p['index']}_{c}") for c in "xyz"])
                self.t0, self.work = context.getParameter("pull_t0"), context.getParameter("pull_work")
            else:  # a new pull from the current positions
                x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                for p in self.vel:
                    p["origin"] = self.com(x, p["beads"]) - (0.0 if p["anchor"] is None else self.com(x, p["anchor"]))
                    for c, value in zip("xyz", p["origin"]):
                        context.setParameter(f"pull_o{p['index']}_{c}", float(value))
                self.t0 = t
                for name, value in (("pull_id", sim.posres_pull_id), ("pull_t0", t), ("pull_work", 0.0), ("pull_time", 0.0)):
                    context.setParameter(name, value)
                note = f"# constant-velocity pull started at time {t:.3f} ps, targets at the current COMs\n"
                print(note[2:].strip())
            for p in self.vel:
                p["force"].setBondParameters(p["bond"], p["groups"], [p["k"], p["v"], *p["origin"], *p["n"]])
            for force in {id(p["force"]): p["force"] for p in self.vel}.values():
                force.updateParametersInContext(context)
            context.setParameter("pull_on", 1.0)
        fname = f"{sim.path}/pull_{sim.config.sysname}.txt"
        if continuing and os.path.isfile(fname):
            # drop lines written after the checkpoint (e.g. by a job killed between checkpoints)
            with open(fname) as f:
                lines = [line for line in f if line.startswith("#") or not line.strip()
                         or float(line.split()[0]) <= t + 1e-6]
            with open(fname, "w") as f:
                f.writelines(lines)
            self.file = open(fname, "a")
        else:
            if os.path.isfile(fname):  # keep the log of the previous pull
                stamp = datetime.now().strftime("%Y%d%m_%Hh%Mm%Ss")
                backup = f"{sim.path}/backup_pull_{sim.config.sysname}_{stamp}.txt"
                os.replace(fname, backup)
                print(f"Backing up {fname} to {backup}")
            self.file = open(fname, "w")
            self.file.write("# pulled groups (see posres_*.txt): id type chains k[kJ/mol/nm^2] "
                            "v[nm/ns] f[kJ/mol/nm] nx ny nz\n")
            for i, p in enumerate(self.pulls, 1):
                k = "-" if p["kind"] == "pull" else f"{p['k']:g}"
                v = "-" if p["kind"] == "pull" else f"{1000 * p['v']:g}"
                f = f"{p['f']:g}" if p["kind"] == "pull" else "-"
                self.file.write(f"#   {i} {p['kind']} {p['label']} {k} {v} {f} "
                                + " ".join(f"{x:.4f}" for x in p["n"]) + "\n")
            if note:
                self.file.write(note)
            self.file.write("# time[ps] pull_time[ps] work[kJ/mol], then per group: d[nm] (COM displacement "
                            "along n; velocity: since the start of the pull) f[kJ/mol/nm] (force along n)\n")
            self.file.flush()
        if self.vel:
            self.move_targets(context)

    def com(self, x: np.ndarray, beads: np.ndarray) -> np.ndarray:
        return self.mass[beads] @ x[beads] / self.mass[beads].sum()

    def move_targets(self, context: openmm.Context) -> None:
        """Move the targets to the middle of the next interval; add the work done."""
        groups = {PULL_FORCE_GROUP}
        e0 = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        t = context.getTime().value_in_unit(unit.picosecond)
        context.setParameter("pull_time", t - self.t0 + 0.5 * self.nupdate * self.dt)
        e1 = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        self.work += (e1 - e0).value_in_unit(unit.kilojoule_per_mole)
        context.setParameter("pull_work", self.work)  # saved in restart.chk

    def describeNextReport(self, simulation: app.Simulation) -> dict[str, Any]:
        step = simulation.currentStep
        to_log = self.nlog - step % self.nlog
        steps = min(to_log, self.nupdate - step % self.nupdate) if self.vel else to_log
        return {"steps": steps, "periodic": False, "include": ["positions"] if steps == to_log else []}

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        step = simulation.currentStep
        if step % self.nlog == 0:
            x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            tp = state.getParameters()["pull_time"] if self.vel else 0.0
            cols = [f"{state.getTime().value_in_unit(unit.picosecond):.3f}", f"{tp:.3f}", f"{self.work:.4f}"]
            for p in self.pulls:
                ref = p["origin"] if p["kind"] == "velocity" else p["target"]
                d = p["n"] @ (self.com(x, p["beads"]) - ref - (0 if p["anchor"] is None else self.com(x, p["anchor"])))
                f = p["f"] if p["kind"] == "pull" else -p["k"] * (d - p["v"] * tp)
                cols += [f"{d:.4f}", f"{f:.3f}"]
            self.file.write(" ".join(cols) + "\n")
            self.file.flush()
        if self.vel and step % self.nupdate == 0:
            self.move_targets(simulation.context)


def add_reporter(sim: Sim, simulation: app.Simulation, append: bool) -> None:
    """Attach PullReporter if the system has pulled groups."""
    if getattr(sim, "posres_pulls", None):
        simulation.reporters.append(PullReporter(sim, simulation, append))

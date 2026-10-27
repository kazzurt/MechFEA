# MechFEA

Windows structural-mechanics research workflow, extracted from KurtsFEM. Includes
quadratic meshing, CalculiX execution, explicit retained-tightening pretension,
contact/load bookkeeping, provenance, numerical acceptance gates, field readers,
comparison tools and a structural desktop interface. Python import name: `structural`.

**Preview:** the assembly preparation is still specialized to the original Dodec
study. This is not a general-purpose CAD-to-FEA product or a certified design-rating
tool. Solver completion and numerical acceptance remain separate states. Private
CAD, COMSOL references and large study results are not included.

## Windows setup

Install Python 3.12 x64 and uv, then from this checkout:

```powershell
.\setup.ps1 -Python C:\path\to\python.exe
.\.venv\Scripts\python.exe -m structural --help
.\.venv\Scripts\python.exe -m pytest -q
```

`setup.ps1` installs the pinned scientific environment and this checkout editable.
For native solves, run `structural-fea-environment/solver/install-solver.ps1`.
It downloads the original pinned Windows archive and verifies archive, executable
and runtime-library hashes. Set `MECHFEA_CCX` to use another explicit installation.
The tested solver is CalculiX 2.23; changing it requires fresh numerical validation.

```powershell
.\.venv\Scripts\python.exe -m structural fixtures --output work/analytical-fixtures
.\.venv\Scripts\python.exe -m structural gui
```

The first command generates synthetic tension, bending, contact and preload cases.
It needs the native solver. It creates fresh outputs and retains numerical evidence.
The GUI's study-specific workflow requires your own prepared project and evidence.
Set `MECHFEA_REFERENCE_DIR` for external reference fields. Coordinates use N, mm, MPa.

FEMview depends only on the lightweight reader/tensor portion of this package;
installing MechFEA without the desktop extra does not require a solver executable.

## Tests and scope

The suite contains synthetic numerical and provenance regressions plus explicit
optional checks against retained external study fixtures. Missing private fixture
checks skip; their assertions are retained. A portable test pass does not establish
mesh convergence, friction qualification or apparatus load capacity. See
`docs/PUBLICATION.md` for extraction and validation details.

Original source: KurtsFEM structural package. Companion projects:
[FEMview](https://github.com/kazzurt/FEMview),
[EMFEA](https://github.com/kazzurt/EMFEA),
[KurtShape](https://github.com/kazzurt/KurtShape).

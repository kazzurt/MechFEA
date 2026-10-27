"""Load Gmsh before other native CAD/Qt libraries in this Windows test process.

Gmsh's bundled OpenCascade DLLs conflict when loaded after OCP/Qt in this pinned
environment. Production meshing uses a separate process with this same order.
"""
import gmsh  # noqa: F401 -- native library load ordering is intentional

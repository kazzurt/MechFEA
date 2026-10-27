"""Structural solid connectivity and face maps, independent of EM meshing.

C3D10 node/face numbering follows the CalculiX 2.23 manual (local downloaded
documentation nodes 34/209/267). Gmsh's last two tetra10 edge nodes are swapped.
Face orders below are outward for a positively oriented tetrahedron; CalculiX
face labels are preserved regardless of this orientation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np

GMSH_TO_CCX_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])
TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
TET10_FACES = {
    1: (0, 2, 1, 6, 5, 4),
    2: (0, 1, 3, 4, 8, 7),
    3: (1, 2, 3, 5, 9, 8),
    4: (2, 0, 3, 6, 7, 9),
}


def mesh_fingerprint(mesh) -> str:
    """Hash physical mesh arrays and named topology, independent of ZIP timestamps.

    Provenance is deliberately excluded so a repair audit can bind this digest
    without a self-referential file hash. Geometry revision has its own verified
    semantic identity; identical arrays alone do not establish revision identity.
    """
    digest=hashlib.sha256()
    for name,values,dtype in (("nodes",mesh.node_ids,"<i8"),("points",mesh.points_mm,"<f8"),
                              ("elements",mesh.element_ids,"<i8"),("tetra10",mesh.tetra10,"<i8")):
        array=np.ascontiguousarray(values,dtype=dtype)
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    topology={"elements":{k:sorted(v) for k,v in mesh.element_sets.items()},
              "nodes":{k:sorted(v) for k,v in mesh.node_sets.items()},
              "surfaces":{k:sorted(map(list,v)) for k,v in mesh.surfaces.items()}}
    digest.update(json.dumps(topology,sort_keys=True,separators=(",",":")).encode())
    return digest.hexdigest()


def verify_gmsh_tet10_order() -> None:
    import gmsh
    _, dimension, order, count, coordinates, _ = gmsh.model.mesh.getElementProperties(11)
    q = np.asarray(coordinates).reshape(-1, 3)[GMSH_TO_CCX_TET10]
    if (dimension, order, count) != (3, 2, 10):
        raise ValueError("Unsupported Gmsh element signature")
    for i, (a, b) in enumerate(TET10_EDGES, 4):
        if not np.allclose(q[i], (q[a]+q[b])/2, atol=1e-12):
            raise ValueError("Gmsh/CalculiX tetra10 edge mapping changed")


@dataclass
class SolidMesh:
    node_ids: np.ndarray
    points_mm: np.ndarray
    element_ids: np.ndarray
    tetra10: np.ndarray  # Actual node IDs, in CalculiX order, not zero-based indices.
    element_sets: dict[str, list[int]] = field(default_factory=dict)
    surfaces: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    node_sets: dict[str, list[int]] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        self.points_mm = np.asarray(self.points_mm, dtype=float)
        self.element_ids = np.asarray(self.element_ids, dtype=np.int64)
        self.tetra10 = np.asarray(self.tetra10, dtype=np.int64)
        self._nodes = {int(n): i for i, n in enumerate(self.node_ids)}
        self._elements = {int(e): i for i, e in enumerate(self.element_ids)}

    def validate(self):
        if self.points_mm.shape != (len(self.node_ids), 3) or self.tetra10.shape != (len(self.element_ids), 10):
            raise ValueError("Mesh node/element array shape mismatch")
        if not np.isfinite(self.points_mm).all():
            raise ValueError("Nonfinite coordinates")
        if len(self._nodes) != len(self.node_ids) or len(self._elements) != len(self.element_ids):
            raise ValueError("Duplicate node or element IDs")
        if np.any(self.node_ids <= 0) or np.any(self.element_ids <= 0):
            raise ValueError("Solver IDs must be positive")
        if not np.isin(self.tetra10, self.node_ids).all():
            raise ValueError("Element references unknown nodes")
        q = self.points_mm[self.connectivity_indices()[:, :4]]
        det = np.linalg.det(np.stack((q[:,1]-q[:,0], q[:,2]-q[:,0], q[:,3]-q[:,0]), axis=-1))
        if not np.all(det > 0):
            raise ValueError("Inverted or degenerate corner tetrahedra")
        quality = self.quadratic_quality()
        if quality["inverted_or_degenerate_elements"]:
            raise ValueError(f"{quality['inverted_or_degenerate_elements']} quadratic tetrahedra have nonpositive sampled Jacobian")
        for name, elements in self.element_sets.items():
            if not elements or not np.isin(elements, self.element_ids).all():
                raise ValueError(f"Empty/invalid element set: {name}")
        for name, surface in self.surfaces.items():
            if not surface or len(set(map(tuple, surface))) != len(surface):
                raise ValueError(f"Empty/repeated faces in surface: {name}")
            for element, face in surface:
                if element not in self._elements or face not in TET10_FACES:
                    raise ValueError(f"Invalid surface reference: {name}")
        for name, nodes in self.node_sets.items():
            if not nodes or not np.isin(nodes, self.node_ids).all():
                raise ValueError(f"Empty/invalid node set: {name}")

    def connectivity_indices(self):
        return np.array([[self._nodes[int(n)] for n in row] for row in self.tetra10], dtype=np.int64)

    def quadratic_quality(self) -> dict:
        """Check integration points, corners, edge mids and centroid in bounded memory.

        Positive sampled Jacobians are necessary, not a proof of positivity at
        every point. Gmsh's native minimum-Jacobian audit remains in provenance.
        """
        a,b=.5854101966249685,.1381966011250105
        barycentric=[np.roll([a,b,b,b],i) for i in range(4)]
        barycentric.extend(np.eye(4))
        barycentric.extend((np.eye(4)[i]+np.eye(4)[j])/2 for i,j in TET10_EDGES)
        barycentric.append(np.full(4,.25))
        dl=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        derivatives=[]
        for l in barycentric:
            derivatives.append(np.vstack(((4*l-1)[:,None]*dl,[4*(l[i]*dl[j]+l[j]*dl[i]) for i,j in TET10_EDGES])))
        indices=self.connectivity_indices()
        bad=0;minimum=float("inf")
        for start in range(0,len(indices),10000):
            points=self.points_mm[indices[start:start+10000]]
            block_min=np.full(len(points),np.inf)
            for derivative in derivatives:
                jac=np.einsum("eni,nj->eij",points,derivative)
                block_min=np.minimum(block_min,np.linalg.det(jac))
            bad+=int(np.count_nonzero(block_min<=0))
            if len(block_min):minimum=min(minimum,float(block_min.min()))
        return {"minimum_sampled_jacobian_mm3":minimum,"inverted_or_degenerate_elements":bad,"samples_per_element":len(derivatives)}

    def face_nodes(self, element: int, face: int) -> np.ndarray:
        return self.tetra10[self._elements[int(element)], list(TET10_FACES[int(face)])]

    def face_points(self, element: int, face: int) -> np.ndarray:
        return self.points_mm[[self._nodes[int(n)] for n in self.face_nodes(element, face)]]

    def surface_nodes(self, name: str) -> list[int]:
        return sorted({int(n) for e, f in self.surfaces[name] for n in self.face_nodes(e, f)})

    def face_lookup(self) -> dict[tuple[int, int, int], list[tuple[int, int]]]:
        lookup = {}
        for eid, row in zip(self.element_ids, self.tetra10):
            for face, nodes in TET10_FACES.items():
                key = tuple(sorted(int(row[n]) for n in nodes[:3]))
                lookup.setdefault(key, []).append((int(eid), face))
        return lookup

    def save(self, path: str | Path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(element_sets=self.element_sets, surfaces=self.surfaces, node_sets=self.node_sets, provenance=self.provenance)
        np.savez_compressed(path, node_ids=self.node_ids, points_mm=self.points_mm,
                            element_ids=self.element_ids, tetra10=self.tetra10,
                            metadata_json=np.array(json.dumps(metadata, allow_nan=False)))

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            mesh = cls(data["node_ids"], data["points_mm"], data["element_ids"], data["tetra10"], **meta)
        mesh.validate()
        return mesh

    def deck_geometry(self) -> str:
        self.validate()
        lines = ["*NODE,NSET=ALL_NODES"]
        lines.extend(f"{int(n)}," + ",".join(f"{x:.14g}" for x in p) for n,p in zip(self.node_ids, self.points_mm))
        lines.append("*ELEMENT,TYPE=C3D10,ELSET=ALL_ELEMENTS")
        lines.extend(f"{int(e)}," + ",".join(str(int(n)) for n in row) for e,row in zip(self.element_ids, self.tetra10))
        for kind, values in (("ELSET", self.element_sets), ("NSET", self.node_sets)):
            for name, ids in values.items():
                if not name.replace("_", "").isalnum():
                    raise ValueError(f"Unsafe solver set name {name}")
                lines.append(f"*{kind},{kind}={name.upper()}")
                lines.extend(",".join(str(int(n)) for n in ids[i:i+16]) for i in range(0,len(ids),16))
        for name, faces in self.surfaces.items():
            if not name.replace("_", "").isalnum():
                raise ValueError(f"Unsafe solver surface name {name}")
            lines.append(f"*SURFACE,NAME={name.upper()},TYPE=ELEMENT")
            lines.extend(f"{e},S{f}" for e,f in faces)
        return "\n".join(lines) + "\n"

    def export_vtu(self, path: str | Path):
        import meshio
        # VTK tetra10 follows the same corner/edge convention as CalculiX.
        body_id = np.zeros(len(self.element_ids), dtype=int)
        names = {}
        for index,(name, ids) in enumerate(self.element_sets.items(), 1):
            names[str(index)] = name
            body_id[np.isin(self.element_ids, ids)] = index
        meshio.write(path, meshio.Mesh(self.points_mm, [("tetra10", self.connectivity_indices())],
            point_data={"solver_node_id": self.node_ids},
            cell_data={"solver_element_id": [self.element_ids], "component_id": [body_id]}))
        Path(str(path)+".components.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

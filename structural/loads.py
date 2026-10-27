"""Specified-direction traction on curved quadratic faces, with force/moment audit."""
from __future__ import annotations
import math
import numpy as np
from .mesh import SolidMesh


def projected_patch_area(radius_mm: float, width_mm: float, axial_length_mm: float) -> float:
    if not 0 < width_mm < 2 * radius_mm or axial_length_mm <= 0:
        raise ValueError("Patch dimensions must fit the cylinder")
    return 2 * radius_mm * math.asin(width_mm / (2 * radius_mm)) * axial_length_mm


def triangle6_shapes(r: float, s: float):
    a = 1-r-s
    values = np.array([a*(2*a-1), r*(2*r-1), s*(2*s-1), 4*a*r, 4*r*s, 4*s*a])
    dr = np.array([1-4*a, 4*r-1, 0, 4*(a-r), 4*s, -4*s])
    ds = np.array([1-4*a, 0, 4*s-1, -4*r, 4*r, 4*(a-s)])
    return values, dr, ds


def triangle6_quadrature(order=6):
    """Tensor Gauss quadrature via a Duffy map to the unit reference triangle."""
    x,w = np.polynomial.legendre.leggauss(order)
    x,w = (x+1)/2,w/2
    for i,r in enumerate(x):
        for j,v in enumerate(x):
            yield r, (1-r)*v, w[i]*w[j]*(1-r)


def consistent_surface_force(mesh: SolidMesh, surface: str, resultant_n, *, moment_origin_mm=(0,0,0), quadrature_order=6):
    """Uniform vector traction per actual curved area, not normal pressure.

    Geometry must partition the footprint boundary first. Selecting whole faces
    by centroid at an unpartitioned patch would alter its area and is forbidden.
    Negative corner weights of curved quadratic triangles are retained.
    """
    target = np.asarray(resultant_n, dtype=float)
    origin = np.asarray(moment_origin_mm, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("A finite 3-vector resultant is required")
    faces = mesh.surfaces.get(surface)
    if not faces or len(set(map(tuple,faces))) != len(faces):
        raise ValueError("Load surface must contain unique, partitioned faces")
    weights = {}
    area = 0.0
    area_centroid_integral = np.zeros(3)
    for element,face in faces:
        points = mesh.face_points(element,face)
        nodes = mesh.face_nodes(element,face)
        nodal_weights = np.zeros(6)
        for r,s,w in triangle6_quadrature(quadrature_order):
            n,dr,ds = triangle6_shapes(r,s)
            jacobian = np.linalg.norm(np.cross(dr @ points, ds @ points))
            if jacobian <= 0 or not np.isfinite(jacobian):
                raise ValueError("Degenerate curved load face")
            da = w * jacobian
            nodal_weights += n * da
            area += da
            area_centroid_integral += da * (n @ points)
        for node,weight in zip(nodes,nodal_weights):
            weights[int(node)] = weights.get(int(node),0.0) + float(weight)
    traction = target/area
    forces = {node: weight*traction for node,weight in weights.items()}
    resultant = sum(forces.values(), np.zeros(3))
    moment = sum((np.cross(mesh.points_mm[mesh._nodes[node]]-origin,force) for node,force in forces.items()), np.zeros(3))
    centroid = area_centroid_integral/area
    expected_moment = np.cross(centroid-origin,target)
    if not np.allclose(resultant,target,rtol=1e-11,atol=1e-10) or not np.allclose(moment,expected_moment,rtol=1e-10,atol=1e-8):
        raise ValueError("Consistent traction resultant/moment check failed")
    report = {"area_mm2": float(area), "centroid_mm": centroid.tolist(), "traction_mpa": traction.tolist(),
              "requested_resultant_n": target.tolist(), "integrated_resultant_n": resultant.tolist(),
              "moment_origin_mm": origin.tolist(), "integrated_moment_nmm": moment.tolist(),
              "integration": f"Quadratic isoparametric face, {quadrature_order}x{quadrature_order} Duffy-Gauss"}
    return forces, report


def cload_text(forces: dict[int, np.ndarray]) -> str:
    lines = ["*CLOAD"]
    for node,force in sorted(forces.items()):
        for dof,value in enumerate(force,1):
            if abs(value) > 1e-14:
                lines.append(f"{node},{dof},{value:.14g}")
    return "\n".join(lines)+"\n" if len(lines)>1 else ""

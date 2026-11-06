"""Independent geometry QA for the new approximate carrier visual reproduction."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "photo_carrier_visual_profile", ROOT / "scripts/photo_carrier_visual_profile.py"
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("manifold3d")
    original_hash = hashlib.sha256(profile.DEFAULT_ORIGINAL.read_bytes()).hexdigest()
    output = tmp_path_factory.mktemp("carrier-visual-qa")
    report = profile.build_carrier_visual_profile(output_dir=output)
    assert hashlib.sha256(profile.DEFAULT_ORIGINAL.read_bytes()).hexdigest() == original_hash
    mesh = trimesh.load_mesh(report["mesh_path"], process=True)
    mesh.apply_scale(1000)  # Explicit metre-to-mm conversion for dimensional QA.
    return report, mesh, output


def circles_at(mesh, z):
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    circles = []
    for loop in section.discrete:
        xy = loop[:-1, :2]
        fit = np.linalg.lstsq(
            np.c_[2 * xy, np.ones(len(xy))], (xy * xy).sum(axis=1), rcond=None
        )[0]
        center = fit[:2]
        radius = np.sqrt(fit[2] + center @ center)
        error = np.max(np.abs(np.linalg.norm(xy - center, axis=1) - radius))
        circles.append((center, radius, error))
    return circles


def vertical_surface_hits(mesh, xy, zmin=14.5, zmax=25):
    """Intersect vertical line with triangle interiors, without optional rtree."""
    tri = mesh.triangles
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    u, v = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    q = np.asarray(xy) - a[:, :2]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(det) > 1e-10
    t = np.zeros(len(tri))
    s = np.zeros(len(tri))
    t[valid] = (q[valid, 0] * v[valid, 1] - q[valid, 1] * v[valid, 0]) / det[valid]
    s[valid] = (u[valid, 0] * q[valid, 1] - u[valid, 1] * q[valid, 0]) / det[valid]
    z = a[:, 2] + t * (b[:, 2] - a[:, 2]) + s * (c[:, 2] - a[:, 2])
    valid &= (t >= -1e-7) & (s >= -1e-7) & (t + s <= 1 + 1e-7)
    return z[valid & (z >= zmin) & (z <= zmax)]


def test_documented_scale_and_provenance(built):
    report, mesh, _ = built
    assert report["newly_reproduced"] is True
    assert report["original_recovered_mesh"] is False
    assert mesh.is_watertight and mesh.is_winding_consistent
    np.testing.assert_allclose(mesh.bounds[:, :2], [[-26.49, -26.49], [26.49, 26.49]], atol=.002)
    np.testing.assert_allclose(mesh.bounds[:, 2], [0, 21.25], atol=.002)
    assert report["upper_plate_mouth_z_mm"] == 17.25
    assert "not recovered corrected STEP or manufacturing CAD" in " ".join(report["limitations"])


def test_section_one_mm_below_mouth_has_expected_openings(built):
    _, mesh, _ = built
    circles = circles_at(mesh, 16.25)
    assert len(circles) == 14  # Exterior, central, 4 insertion, 4 tiny, 4 mounting.
    expected = [(np.array([0., 0.]), 11.475)]
    for angle in np.arange(4) * np.pi / 2:
        expected.append((17.9625 * np.array([np.cos(angle), np.sin(angle)]), 4.12))
        phase = angle - np.pi / 8
        expected.append((13.37 * np.array([np.cos(phase), np.sin(phase)]), 1.3))
    for center, radius in expected:
        # The exterior and central bore are concentric; match radius as well.
        candidates = [item for item in circles if abs(item[1] - radius) < .5]
        assert candidates
        found = min(candidates, key=lambda item: np.linalg.norm(item[0] - center))
        np.testing.assert_allclose(found[0], center, atol=.015)
        assert abs(found[1] - radius) < .02
        assert found[2] < .025


def test_open_bores_have_no_coincident_top_cap_or_clog(built):
    _, mesh, _ = built
    # These are geometry rays, independent of the metadata's reported centers.
    expected = [(np.array([0., 0.]), 11.475)]
    for angle in np.arange(4) * np.pi / 2:
        expected.append((17.9625 * np.array([np.cos(angle), np.sin(angle)]), 4.12))
        phase = angle - np.pi / 8
        expected.append((13.37 * np.array([np.cos(phase), np.sin(phase)]), 1.3))
    for center, radius in expected:
        assert len(vertical_surface_hits(mesh, center)) == 0
        for angle in np.arange(12) * np.pi / 6:
            direction = np.array([np.cos(angle), np.sin(angle)])
            assert len(vertical_surface_hits(mesh, center + (radius - .05) * direction)) == 0
            hits = vertical_surface_hits(mesh, center + (radius + .05) * direction)
            assert np.any(np.isclose(hits, 17.25, atol=.002))
    # Above the upper plate only four raised bosses and their holes remain.
    assert len(circles_at(mesh, 18.25)) == 8


def test_adapter_preserves_existing_assets_and_nonvisual_nodes(built):
    report, _, output = built
    root = ET.fromstring("""<mujoco><asset>
      <mesh name="large_carrier_visual" file="original.stl" scale="1 1 1"/>
      </asset><worldbody><body name="carrier"><inertial mass="1"/>
      <geom name="large_carrier_visual" mesh="large_carrier_visual" contype="0" conaffinity="0" density="0"/>
      <geom name="physical" type="box" contype="1" conaffinity="1" size="1 2 3"/>
      <site name="insertion_target" pos=".0179625 0 .01725"/>
      </body></worldbody></mujoco>""")
    original_asset = ET.tostring(root.find("asset/mesh"))
    before = [ET.tostring(root.find(path)) for path in (
        "worldbody/body/inertial", "worldbody/body/geom[@name='physical']", "worldbody/body/site"
    )]
    profile.apply_carrier_visual_profile(root, artifacts={"output_dir": output})
    profile.apply_carrier_visual_profile(root, artifacts={"output_dir": output})
    assert ET.tostring(root.find("asset/mesh[@name='large_carrier_visual']")) == original_asset
    after = [ET.tostring(root.find(path)) for path in (
        "worldbody/body/inertial", "worldbody/body/geom[@name='physical']", "worldbody/body/site"
    )]
    assert before == after
    assert len(root.findall("asset/mesh[@name='photo_reproduced_carrier_real_face_v4']")) == 1
    assert root.find("worldbody/body/geom[@name='large_carrier_visual']").get("mesh") == "photo_reproduced_carrier_real_face_v4"
    assert report["physics_modified"] is False

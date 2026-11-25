import numpy as np

from lesionagent.core.measure import measure, slice_axis


def ellipse(shape=(20, 200, 200), sp=(2.5, 0.7, 0.7), a=20.0, b=10.0, th=30, c=(25, 70, 70)):
    z, y, x = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, sp)], indexing="ij")
    t = np.deg2rad(th)
    xr = (x - c[2]) * np.cos(t) + (y - c[1]) * np.sin(t)
    yr = -(x - c[2]) * np.sin(t) + (y - c[1]) * np.cos(t)
    return (xr / a) ** 2 + (yr / b) ** 2 + ((z - c[0]) / 10) ** 2 <= 1, sp


def test_rotated_ellipse_axial():
    m, sp = ellipse()
    r = measure(m, sp)
    assert r.slice_axis == 0
    assert abs(r.ld_mm - 40) < 1.5 and abs(r.sa_mm - 20) < 1.5
    assert r.ld_slice == 10 and r.ld_line is not None and r.ld_line[0][0] == 10
    # endpoints are ~ld apart in mm
    p, q = np.array(r.ld_line)
    assert abs(np.linalg.norm((p - q) * sp) - r.ld_mm) < 0.5


def test_slice_axis_from_direction():
    assert slice_axis(np.eye(3).ravel()) == 0                     # axial: ITK z = numpy axis0
    d = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)     # ITK y index runs S-I (coronal stack)
    assert slice_axis(d.ravel()) == 1


def test_empty():
    assert measure(np.zeros((5, 5, 5), bool), (1, 1, 1)).n_voxels == 0

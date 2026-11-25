"""Views keep >= VIEW_MIN_MM of context (shifted inwards at the scan edge); accepted candidates made of several
separate components become one lesion per component (>= SPLIT_MIN_ML), smaller fragments are dropped."""
import numpy as np

from lesionagent.core import rules
from lesionagent.workspace import RecistWorkspace


class _Vol:
    def __init__(self, shape, spacing):
        self.shape, self.spacing = shape, spacing


def _ws(shape, spacing):
    ws = object.__new__(RecistWorkspace)
    ws._vol = lambda v: _Vol(shape, spacing)
    return ws


def test_fov_min_context_and_edge_shift():
    ws = _ws((100, 512, 512), (2.5, 0.7, 0.7))
    fov = ws._fov(None, [2, 250, 250], None, [[0, 5], [240, 260], [240, 260]])
    assert fov[0][0] == 0 and (fov[0][1] - fov[0][0]) * 2.5 >= rules.VIEW_MIN_MM  # shifted, not cut at z = 0
    assert (fov[1][1] - fov[1][0]) * 0.7 >= rules.VIEW_MIN_MM


def test_split_parts():
    m = np.zeros((40, 80, 80), bool)
    m[5:15, 10:30, 10:30] = True    # 4 ml
    m[20:30, 50:70, 50:70] = True   # 4 ml
    m[35, 5, 5] = True              # fragment
    ws = _ws(m.shape, (1.0, 1.0, 1.0))
    parts, dropped = ws._split_parts(None, m)
    assert [int(p.sum()) for p in parts] == [4000, 4000] and dropped == 1
    one = np.zeros_like(m)
    one[5:15, 10:30, 10:30] = True
    one[35, 5, 5] = True            # one lesion + a fragment: unchanged
    parts, dropped = ws._split_parts(None, one)
    assert len(parts) == 1 and parts[0] is one and dropped == 0

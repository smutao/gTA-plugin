"""
generalized Turnstile Assistant (gTA)

PyMOL plugin that performs turnstile rotations on molecular structures.

Key behavior:
- User selects one anchor atom, then one connection atom for each arm.
- After selecting each arm connection atom, press Wizard panel button
  'Arm Atoms Selection Done' to record that arm.
- When done selecting arms (arm count >= 2), click GUI 'Picking Finished' to
  compute the rotation axis and enable the angle slider to rotate the selected
  fragment(s) in-place by updating atomic coordinates via cmd.alter_state.

Environment constraints:
- PyMOL v3.0 (tested), only 1 object with 1 state is loaded.
- NumPy / SciPy allowed; RDKit is not allowed.
- Bonds topology comes from PyMOL get_model; no bond inference is performed.

Author: Yunwen Tao, Ph.D.
Date: 2025-10-06
"""

from __future__ import absolute_import, print_function

import os
import math
import numpy as np
from scipy.optimize import minimize

import pymol
from pymol import cmd
from pymol.wizard import Wizard


# =============================
# Math / Geometry (from gTA-cli)
# =============================

def _R(theta, u):
    """Rotation matrix for angle theta around unit vector u.
    Args:
        theta (float): angle in radians
        u (array-like): 3-vector unit axis
    Returns:
        np.ndarray of shape (3,3)
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    omct = 1.0 - ct
    ux, uy, uz = u

    ux_uy = ux * uy
    ux_uz = ux * uz
    uy_uz = uy * uz

    return np.array([
        [ct + ux*ux * omct,      ux_uy * omct - uz*st,  ux_uz * omct + uy*st],
        [ux_uy * omct + uz*st,   ct + uy*uy * omct,     uy_uz * omct - ux*st],
        [ux_uz * omct - uy*st,   uy_uz * omct + ux*st,  ct + uz*uz * omct]
    ])


def _Rotate3(anchor, point_to_rotate, axis_point, theta):
    """Rotate a point around axis defined by (anchor -> axis_point).
    Args:
        anchor (array-like 3, np.ndarray preferred)
        point_to_rotate (array-like 3)
        axis_point (array-like 3)
        theta (float): radians
    Returns:
        np.ndarray rotated point
    """
    if not isinstance(anchor, np.ndarray):
        anchor = np.array(anchor)
    if not isinstance(axis_point, np.ndarray):
        axis_point = np.array(axis_point)
    if not isinstance(point_to_rotate, np.ndarray):
        point_to_rotate = np.array(point_to_rotate)

    u = axis_point - anchor
    u = u / np.linalg.norm(u)

    r = _R(theta, u)
    relative = point_to_rotate - anchor
    rotated = r @ relative + anchor
    return rotated


def _get_polygon_points(norm, center, r, num_gon, phi):
    """Compute regular polygon vertices on plane defined by normal & center."""
    norm = np.array(norm, dtype=float)
    center = np.array(center, dtype=float)
    # Create two orthonormal vectors on the plane
    v1 = np.array([-norm[1], norm[0], 0.0], dtype=float)
    if np.linalg.norm(v1) < 1e-12:
        # If norm is near z-axis, choose different perpendicular
        v1 = np.array([1.0, 0.0, 0.0], dtype=float)
    v2 = np.cross(norm, v1)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    points = []
    for i in range(num_gon):
        angle = i * 2.0 * math.pi / num_gon + phi
        vec = math.cos(angle) * v1 + math.sin(angle) * v2
        points.append(center + r * vec)
    return np.array(points)


def _determine_rotation_direction(points, center_point, normal):
    """Determine rotation direction (+1 or -1) based on projected polygon area."""
    points = np.array(points, dtype=float)
    center_point = np.array(center_point, dtype=float)
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    v1 = np.array([-normal[1], normal[0], 0.0], dtype=float)
    if np.linalg.norm(v1) < 1e-12:
        v1 = np.array([1.0, 0.0, 0.0], dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    projected = []
    for p in points:
        rel = p - center_point
        x = np.dot(rel, v1)
        y = np.dot(rel, v2)
        projected.append([x, y])
    projected = np.array(projected)

    total_cross = 0.0
    n = len(projected)
    for i in range(n):
        p1 = projected[i]
        p2 = projected[(i + 1) % n]
        cross = np.cross(p1, p2)
        total_cross += cross

    # If cross sum positive: counterclockwise → return -1 to match CLI behavior
    return -1 if total_cross > 0 else 1


def _calc_deviation(params, anchor_point, sphere_points, direction):
    """Objective function to fit polygon to reference points on sphere."""
    circle_center = params[0:3]
    r = params[3]
    phi = params[4]

    norm = circle_center - anchor_point
    norm = norm / np.linalg.norm(norm)
    norm = -norm * direction

    num_arm = len(sphere_points)
    polygon_points = _get_polygon_points(norm, circle_center, r, num_arm, phi)

    # Sum squared distances to target points
    deviation = 0.0
    for i in range(num_arm):
        dist = np.linalg.norm(polygon_points[i] - sphere_points[i])
        deviation += dist * dist
    return deviation


def _get_optimized_points(circle_center, r, phi, anchor_point, num_arm, direction):
    norm = circle_center - anchor_point
    norm = norm / np.linalg.norm(norm)
    norm = -norm * direction
    return _get_polygon_points(norm, circle_center, r, num_arm, phi)


def _obtain_new_circle_center(anchor_point, first_shell_arm_points, reverse_order=True):
    """Compute circle center based on anchor and first-shell arm points.

    This matches the logic in gTA-cli, without any file I/O.
    """
    anchor_point = np.array(anchor_point, dtype=float)
    first_shell_arm_points = np.array(first_shell_arm_points, dtype=float)

    if reverse_order:
        first_shell_arm_points = first_shell_arm_points[::-1]

    num_arm = len(first_shell_arm_points)
    assert num_arm >= 2

    # Project arm points onto reference sphere around anchor
    reference_radius = 2.0
    sphere_points = []
    for i in range(num_arm):
        dist = np.linalg.norm(first_shell_arm_points[i] - anchor_point)
        sp = anchor_point + (first_shell_arm_points[i] - anchor_point) / dist * reference_radius
        sphere_points.append(sp)
    sphere_points = np.array(sphere_points)

    if num_arm == 2:
        return (sphere_points[0] + sphere_points[1]) / 2.0

    sphere_center = np.mean(sphere_points, axis=0)
    r0 = np.linalg.norm(sphere_center - sphere_points[0])
    norm0 = sphere_center - anchor_point
    norm0 = norm0 / np.linalg.norm(norm0)

    direction = _determine_rotation_direction(
        points=first_shell_arm_points,
        center_point=anchor_point,
        normal=norm0,
    )

    initial_guess = np.concatenate([sphere_center, [r0, 0.0]])
    margin = 4.0
    bounds = [
        (sphere_center[0] - margin, sphere_center[0] + margin),
        (sphere_center[1] - margin, sphere_center[1] + margin),
        (sphere_center[2] - margin, sphere_center[2] + margin),
        (0.5 * r0, 2.5 * r0),
        (-2.0 * math.pi, 2.0 * math.pi),
    ]

    fn = lambda p: _calc_deviation(p, anchor_point=anchor_point, sphere_points=sphere_points, direction=direction)
    result = minimize(fn, initial_guess, bounds=bounds)
    return result.x[0:3]


# =============================
# Graph Utilities (from gTA-cli logic; no RDKit)
# =============================

def _find_connected_atoms(adj_matrix, start_atoms):
    """BFS over adjacency matrix, starting from multiple nodes.
    Args:
        adj_matrix (np.ndarray NxN): binary adjacency
        start_atoms (list[int]): zero-based indices
    Returns:
        sorted list of zero-based indices visited
    """
    visited = set(start_atoms)
    queue = list(start_atoms)
    n = adj_matrix.shape[0]
    while queue:
        current = queue.pop(0)
        # Iterate neighbors
        for neighbor in range(n):
            if adj_matrix[current, neighbor] == 1 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return sorted(list(visited))


# =============================
# Wizard for picking atoms
# =============================

object_prefix = "_pw"
object_subgroup_prefix = "_s"


class GTAWizard(Wizard):
    """Wizard using TATA-style picking flow and selection naming.

    Behavior:
      - First pick is anchor (auto-finish for that group).
      - Then for each arm: pick one or more atoms; press 'Arm Atoms Selection Done'
        in Wizard panel to finish that arm group. For gTA, only the first pick in
        each arm group is used as the arm connection atom.
    """

    def __init__(self):
        Wizard.__init__(self)
        self.pick_count = 0          # number of finished groups (anchor=0th, then arms)
        self.subgroup_count = 0      # picks within current group
        self.subgroup_sum = []       # list of counts per finished group
        self.object_count = 0
        self.object_prefix = object_prefix
        self.object_subgroup_prefix = object_subgroup_prefix
        self.picking_finished = False

        self.selection_mode = cmd.get_setting_legacy("mouse_selection_mode")
        cmd.set("mouse_selection_mode", 0)  # atomic
        cmd.deselect()

    def reset(self):
        cmd.delete(self.object_prefix + "*")
        cmd.delete("_indicate*")
        cmd.unpick()
        self.pick_count = 0
        self.subgroup_count = 0
        self.subgroup_sum = []
        self.picking_finished = False
        cmd.refresh_wizard()

    def cleanup(self):
        cmd.set("mouse_selection_mode", self.selection_mode)
        self.reset()

    def get_prompt(self):
        if self.picking_finished:
            return ['Picking finished. Use GUI slider to rotate, or press Done/Reset.']
        else:   
            if self.pick_count == 0:
                return ['Please click on the anchor atom...']
            else:
                arm_num = self.pick_count  # 1-based arm index
                return [f'Pick arm #{arm_num} link atom, then press "Arm Atoms Selection Done"','OR','Click the "Picking Finished" button']

    def do_select(self, name):
        cmd.edit("%s and not %s*" % (name, self.object_prefix))
        self.do_pick(0)

    def pickNextAtom(self, atom_name):
        cmd.select(atom_name, "(pk1)")
        print(atom_name)
        cmd.unpick()
        indicate_selection = "_indicate" + self.object_prefix
        cmd.select(indicate_selection, atom_name)
        cmd.enable(indicate_selection)
        self.subgroup_count += 1

    def do_pick(self, picked_bond):
        if picked_bond:
            print("Error: please select atoms, not bonds")
            return

        atom_name = (
            self.object_prefix
            + str(self.pick_count)
            + self.object_subgroup_prefix
            + str(self.subgroup_count)
        )

        self.pickNextAtom(atom_name)

        if self.pick_count == 0:
            # Anchor group: auto-finish after first pick
            self.finish_1arm()

    def finish_1arm(self):
        # Finish current group (anchor or one arm)
        self.pick_count += 1
        self.subgroup_sum.append(self.subgroup_count)
        self.subgroup_count = 0
        cmd.refresh_wizard()

    def get_panel(self):
        return [
            [1, 'gTA Turnstile Wizard', ''],
            [2, 'Reset', 'cmd.get_wizard().reset()'],
            [2, 'Arm Atom Selection Done', 'cmd.get_wizard().finish_1arm()'],
            [2, 'Done', 'cmd.set_wizard()'],
        ]


# =============================
# GUI wiring and plugin entry points
# =============================

dialog = None  # keep global reference to prevent GC


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('generalized Turnstile Assistant', run_plugin_gui)


def run_plugin_gui():
    global dialog
    if dialog is None:
        dialog = _make_dialog()
    dialog.show()


def _make_dialog():
    from pymol.Qt import QtWidgets
    from pymol.Qt.utils import loadUi

    # State shared between UI callbacks
    state = {
        'wizard': GTAWizard(),
        'object_name': None,
        'id_to_idx0': None,
        'idx0_to_id': None,
        'coords0': None,           # initial coordinates (np.ndarray Nx3)
        'anchor_idx0': None,
        'arm_idx0_list': None,     # list[int]
        'relevant_idx0': None,     # list[int] to rotate
        'anchor_point': None,
        'axis_point': None,        # new_circle_center
    }

    dlg = QtWidgets.QDialog()
    uifile = os.path.join(os.path.dirname(__file__), 'demowidget.ui')
    form = loadUi(uifile, dlg)

    wiz = state['wizard']

    # Initial UI state
    form.slider_angle.setDisabled(True)
    form.angle_text.setDisabled(True)
    form.set_angle.setDisabled(True)
    # Step size 1 degree as required
    form.slider_angle.setSingleStep(1)

    def _set_status(msg):
        try:
            form.status_text.setText(msg)
        except Exception:
            print(msg)

    def _reset_runtime():
        # Reset runtime-only (post-picking) caches
        state['object_name'] = None
        state['id_to_idx0'] = None
        state['idx0_to_id'] = None
        state['coords0'] = None
        state['anchor_idx0'] = None
        state['arm_idx0_list'] = None
        state['relevant_idx0'] = None
        state['anchor_point'] = None
        state['axis_point'] = None

    def start_wiz():
        wiz.reset()
        cmd.set_wizard(wiz)
        _reset_runtime()

        form.slider_angle.setDisabled(True)
        form.slider_angle.setValue(0)
        form.angle_text.setDisabled(True)
        form.set_angle.setDisabled(True)
        _set_status("Please select anchor then arm connection atoms. \nAfter each arm pick, press 'Arm Atom Selection Done'. \nWhen finished, click 'Picking Finished'.")

    def _build_model_info():
        # Expect exactly one object loaded
        names = cmd.get_names('objects')
        if not names:
            raise RuntimeError('No object loaded. Please load a structure first.')
        obj = names[0]
        md = cmd.get_model(obj)
        coords = np.array(md.get_coord_list(), dtype=float)
        n = len(md.atom)
        idx0_to_id = [at.id for at in md.atom]  # 1-based IDs
        id_to_idx0 = {idx0_to_id[i]: i for i in range(n)}

        # Build adjacency from bonds (bond.index are zero-based pairs)
        adj = np.zeros((n, n), dtype=np.int8)
        for bd in md.bond:
            i, j = bd.index  # 0-based indices in md.atom
            adj[i, j] = 1
            adj[j, i] = 1

        return obj, coords, id_to_idx0, idx0_to_id, adj

    def picking_finish():
        # Validate groups: subgroup_sum contains [anchor_count, arm1_count, arm2_count, ...]
        if len(wiz.subgroup_sum) < 3:
            _set_status("Please finish selection: anchor + at least 2 arms. Use 'Arm Atom Selection Done' after each arm.")
            return
        if wiz.subgroup_sum[0] != 1:
            _set_status("Anchor selection should contain exactly 1 atom. Reset and try again.")
            return

        try:
            obj, coords, id_to_idx0, idx0_to_id, adj = _build_model_info()
        except Exception as e:
            _set_status(f"Error reading model: {e}")
            return

        wiz.picking_finished = True
        cmd.refresh_wizard()


        # Extract anchor id from selection "_pw0_s0"
        str00 = object_prefix + "0" + object_subgroup_prefix + "0"
        md_anchor = cmd.get_model(str00, 1)
        if not md_anchor.atom:
            _set_status("Anchor selection is empty. Reset and try again.")
            return
        anchor_id = md_anchor.atom[0].id
        anchor_idx0 = id_to_idx0.get(anchor_id)
        if anchor_idx0 is None:
            _set_status("Anchor atom not found in object model.")
            return

        # Build arm list using the first picked atom in each arm group: "_pw{i}_s0"
        arm_idx0_list = []
        first_shell_arm_points = []
        for i in range(1, len(wiz.subgroup_sum)):
            sel = object_prefix + str(i) + object_subgroup_prefix + "0"
            md_arm = cmd.get_model(sel, 1)
            if not md_arm.atom:
                _set_status(f"Arm #{i} selection is empty. Reset and try again.")
                return
            arm_id = md_arm.atom[0].id
            idx0 = id_to_idx0.get(arm_id)
            if idx0 is None:
                _set_status(f"Arm #{i} atom not found in object model.")
                return
            arm_idx0_list.append(idx0)
            first_shell_arm_points.append(coords[idx0])

        first_shell_arm_points = np.array(first_shell_arm_points, dtype=float)

        # Determine fragment to rotate: break bonds between anchor and each arm, BFS from all arms
        adj2 = adj.copy()
        for ai in arm_idx0_list:
            adj2[anchor_idx0, ai] = 0
            adj2[ai, anchor_idx0] = 0
        relevant_idx0 = _find_connected_atoms(adj2, arm_idx0_list)

        # Compute axis point (new circle center) using original coordinates
        anchor_point = coords[anchor_idx0]
        axis_point = _obtain_new_circle_center(anchor_point, first_shell_arm_points, reverse_order=True)

        # Cache runtime info for slider-driven updates
        state['object_name'] = obj
        state['id_to_idx0'] = id_to_idx0
        state['idx0_to_id'] = idx0_to_id
        state['coords0'] = coords.copy()
        state['anchor_idx0'] = anchor_idx0
        state['arm_idx0_list'] = arm_idx0_list
        state['relevant_idx0'] = relevant_idx0
        state['anchor_point'] = anchor_point.copy()
        state['axis_point'] = np.array(axis_point, dtype=float)

        # Enable controls
        form.slider_angle.setEnabled(True)
        form.angle_text.setEnabled(True)
        form.set_angle.setEnabled(True)
        form.angle_text.setText(str(form.slider_angle.value()))

        _set_status(f"Selected {len(arm_idx0_list)} arms; rotating {len(relevant_idx0)} atoms")

    def _apply_rotation(deg):
        if state['coords0'] is None:
            return
        theta = math.radians(float(deg))
        anchor = state['anchor_point']
        axis = state['axis_point']
        coords0 = state['coords0']
        idxs = state['relevant_idx0']
        idmap = state['idx0_to_id']

        for i in idxs:
            newp = _Rotate3(anchor, coords0[i], axis, theta)
            x, y, z = newp.tolist()
            cmd.alter_state(1, f"id {idmap[i]}", f"(x,y,z)=({x},{y},{z})")
        cmd.rebuild()

    def slider_move():
        val = form.slider_angle.value()
        form.angle_text.setText(str(val))
        _apply_rotation(val)

    def specify_angle():
        # Parse angle from text and clamp
        try:
            angle = float(form.angle_text.text())
        except Exception:
            angle = 0.0
        angle = max(-180.0, min(180.0, angle))
        form.slider_angle.setValue(int(round(angle)))

    def revert_changes():
        form.angle_text.setText("0")
        specify_angle()

    # Wire up callbacks
    form.button_close.clicked.connect(dlg.close)
    form.start.clicked.connect(start_wiz)
    form.pick_finish.clicked.connect(picking_finish)
    form.slider_angle.valueChanged.connect(slider_move)
    form.set_angle.clicked.connect(specify_angle)
    form.revert.clicked.connect(revert_changes)

    # Initial status
    _set_status("Click Start, then select anchor & arms. \nPress 'Arm Atom Selection Done' in Wizard after each arm. \nFinally, click 'Picking Finished'.")

    return dlg

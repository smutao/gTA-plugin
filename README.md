# generalized Turnstile Assistant (gTA)

`generalized Turnstile Assistant (gTA)` is a PyMOL plugin for performing generalized turnstile rotations on molecular structures directly inside the PyMOL GUI.

## Dependencies

This plugin requires:

- PyMOL 3.0 or later
- `scipy`

## Installing PyMOL

Open-source PyMOL is available from `conda-forge`.

Run the following commands in a terminal:

```bash
conda create -n pymol-opensource
conda activate pymol-opensource
conda install -c conda-forge pymol-open-source
conda install -c conda-forge scipy
pymol
```

## Installing the Plugin

After launching PyMOL on your computer, open the menu bar and go to `Plugin -> Plugin Manager`.

In the `Plugin Manager` dialog:

1. Open the `Install New Plugin` tab.
2. Click `Choose file...`.
3. Navigate to the gTA plugin folder in this repository.
4. Select `__init__.py`.
5. Click `OK` in the remaining pop-up windows.

The plugin is then installed successfully. You should see `generalized Turnstile Assistant` in the `Plugin` menu of PyMOL.

## How to Use

### Workflow

1. Launch PyMOL and load your molecular structure.
2. Open `Plugin -> generalized Turnstile Assistant`.
3. Click `Start`.
4. In the PyMOL viewport, click the central atom.
5. Click one connection atom for the first arm.
6. In the PyMOL wizard panel, click `Arm Atom Selection Done`.
7. Repeat the same process for each additional arm.
8. After selecting at least two arms, click `Picking Finished` in the plugin window.
9. Use the angle slider to rotate the selected fragment.
10. You can also type a value into the angle box and click `Set Angle`.
11. Click `Revert Changes` to return the angle to `0`.

### What the Plugin Does

- The first picked atom is treated as the central atom.
- For each arm, the first picked atom is used as the arm connection atom.
- After picking is completed, the plugin computes the rotation axis automatically.
- Rotation is then applied directly to the relevant atoms in the loaded PyMOL object.

## Current Assumptions

The current implementation assumes:

- only one object is loaded in PyMOL
- the object has only one state
- arm selection requires at least two arms

## Reference

 *Generalized Turnstile Rotation: Formulation, Visualization, Workflow Implementation, and Application for Modeling Polytopal Rearrangements.* ChemRxiv. 15 February 2026. DOI: https://doi.org/10.26434/chemrxiv.15000069/v1

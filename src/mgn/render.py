"""Offscreen PyVista renders with a fixed camera per trajectory, plus world -> pixel projection."""
import numpy as np
import pyvista as pv

from mgn.data import OBSTACLE, T_MAX

pv.OFF_SCREEN = True
SIZE = 448


class Renderer:
    def __init__(self, traj):
        nt, cells = traj["node_type"], traj["cells"]
        is_obs = (nt[cells] == OBSTACLE).all(1)
        self.plate = self._grid(cells[~is_obs], traj["world_pos"][0])
        self.actuator = self._grid(cells[is_obs], traj["world_pos"][0])
        self.p = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE))
        self.p.set_background("white")
        # fixed camera: fit the whole modelled window once, reuse for every frame
        pts = traj["world_pos"][[0, T_MAX - 1]].reshape(-1, 3)
        self.p.add_mesh(pv.PolyData(pts), opacity=0)
        self.p.view_vector((1, 0.4, -0.5), viewup=(0, 0, 1))  # from below: actuator pushes the plate upward
        self.p.reset_camera()
        self.p.camera.zoom(1.05)
        self.camera = self.p.camera.copy()
        self.p.clear()

    @staticmethod
    def _grid(cells, pos):
        c = np.hstack([np.full((len(cells), 1), 4), cells]).ravel()
        return pv.UnstructuredGrid(c, np.full(len(cells), pv.CellType.TETRA), pos.astype(np.float64))

    def render(self, pos, scalars=None, clim=None):
        """RGB image of the deformed plate (grey, or coloured by per-node `scalars`) and the actuator (blue)."""
        self.p.clear()
        self.plate.points = self.actuator.points = pos.astype(np.float64)
        plate = self.plate.extract_surface(algorithm="dataset_surface")
        if scalars is None:
            self.p.add_mesh(plate, color="lightgrey", show_edges=True, edge_color="grey", line_width=0.5)
        else:
            plate["score"] = scalars[plate["vtkOriginalPointIds"]]
            self.p.add_mesh(plate, scalars="score", cmap="YlOrRd", clim=clim, show_scalar_bar=False, show_edges=True,
                            edge_color="grey", line_width=0.3)
        self.p.add_mesh(self.actuator.extract_surface(algorithm="dataset_surface"), color="steelblue")
        self.p.camera = self.camera.copy()
        return self.p.screenshot(return_img=True)

    def project(self, points):
        """World coordinates -> (x, y) pixel coordinates, y down (image convention)."""
        vtk_m = self.camera.GetCompositeProjectionTransformMatrix(1.0, -1, 1)
        m = np.array([[vtk_m.GetElement(i, j) for j in range(4)] for i in range(4)])
        h = np.c_[points, np.ones(len(points))] @ m.T
        ndc = h[:, :2] / h[:, 3:]
        return np.c_[(ndc[:, 0] + 1) / 2 * SIZE, (1 - ndc[:, 1]) / 2 * SIZE]


def grid_cell(xy, n=3):
    """3x3 image grid cell ('top-left' ... 'bottom-right') containing the centroid of pixel coords `xy`."""
    cx, cy = (np.clip(xy.mean(0), 0, SIZE - 1) * n / SIZE).astype(int)
    return f"{['top', 'middle', 'bottom'][cy]}-{['left', 'center', 'right'][cx]}"

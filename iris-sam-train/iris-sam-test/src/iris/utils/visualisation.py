from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np

from iris.io import dataclasses as iris_dc

Canvas = Tuple[matplotlib.figure.Figure, Union[matplotlib.axes._axes.Axes, np.ndarray]]


class IRISVisualizer:
    """IRISPipeline outputs visualizer."""

    def __init__(self) -> None:
        """Assign parameters."""
        self.cyclic_cmap = self._make_cyclic_cmap()
        self.linear_cmap = self._make_linear_cmap()

    def plot_ir_image(
        self, ir_image: Union[iris_dc.IRImage, Dict[str, Any]], ax: Optional[matplotlib.axes._axes.Axes] = None
    ) -> Canvas:
        """Visualise an IRIS IRImage.

        Args:
            ir_image (Union[iris_dc.IRImage, Dict[str, Any]]): input image of iris' IRImage type
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)

        axis.imshow(ir_image.img_data, cmap="gray")
        axis.set_title(ir_image.eye_side)
        axis.set_xlabel(ir_image.width)
        axis.set_ylabel(ir_image.height)

        return fig, axis

    def plot_ir_image_with_landmarks(
        self,
        ir_image: Union[iris_dc.IRImage, Dict[str, Any]],
        landmarks: Union[iris_dc.Landmarks, Dict[str, List[float]]],
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Plot landmarks together with IR image.

        Args:
            ir_image (Union[iris_dc.IRImage, Dict[str, Any]]): IR image.
            landmarks (Union[iris_dc.Landmarks, Dict[str, List[float]]]): Landmarks.
            ax (Optional[matplotlib.axes._axes.Axes], optional): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """

        def loop_back(array):
            return np.concatenate([array, [array[0]]], axis=0)

        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)
        if isinstance(landmarks, dict):
            landmarks = iris_dc.Landmarks.deserialize(landmarks)

        axis.imshow(ir_image.img_data, cmap="gray")
        for lnd, color in [
            ("pupil_landmarks", "#0099FF"),
            ("iris_landmarks", "#FFB900"),
            ("eyeball_landmarks", "#00B979"),
        ]:
            axis.scatter(*getattr(landmarks, lnd).T, marker="+", color=color, s=150)
            axis.plot(
                loop_back(getattr(landmarks, lnd)[:, 0]),
                loop_back(getattr(landmarks, lnd)[:, 1]),
                marker="+",
                color=color,
            )

        return fig, axis

    def plot_segmentation_map(
        self,
        segmap: Union[iris_dc.SegmentationMap, Dict[str, Any]],
        ir_image: Optional[Union[iris_dc.IRImage, Dict[str, Any]]] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Plot segmentation maps.

        Args:
            segmap (Union[iris_dc.SegmentationMap, Dict[str, Any]]): Segmentation maps.
            ir_image (Optional[Union[iris_dc.IRImage, Dict[str, Any]]], optional): IR image. Defaults to None.
            ax (Optional[matplotlib.axes._axes.Axes], optional): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, np.ndarray] = self._init_canvas(ax, subplot_size=(1, 4))
        fig, axs = canvas
        fig.set_figwidth(18)
        fig.set_figheight(16)

        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)
        if isinstance(segmap, dict):
            segmap = iris_dc.SegmentationMap.deserialize(segmap)

        if ir_image is not None:
            axs[0].imshow(ir_image.img_data, cmap="gray")
            axs[1].imshow(ir_image.img_data, cmap="gray")
            axs[2].imshow(ir_image.img_data, cmap="gray")
            axs[3].imshow(ir_image.img_data, cmap="gray")

        axs[0].imshow(segmap.predictions[..., 0], alpha=0.6)
        axs[1].imshow(segmap.predictions[..., 1], alpha=0.6)
        axs[2].imshow(segmap.predictions[..., 2], alpha=0.6)
        axs[3].imshow(segmap.predictions[..., 3], alpha=0.6)

        return fig, axs

    def plot_geometry_mask(
        self,
        geometry_mask: Union[iris_dc.GeometryMask, Dict[str, Any]],
        ir_image: Optional[Union[iris_dc.IRImage, Dict[str, Any]]] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualize an IRIS GeometryMask objet.

        Args:
            geometry_mask (Union[iris_dc.GeometryMask, Dict[str, Any]]): Geometry mask.
            ir_image (Optional[Union[iris_dc.IRImage, Dict[str, Any]]], optional): Optional IRImage to lay over in transparency. Defaults to None.
            ax (Optional[matplotlib.axes._axes.Axes], optional): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(geometry_mask, dict):
            geometry_mask = iris_dc.GeometryMask.deserialize(geometry_mask)
        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)

        if ir_image is not None:
            axis.imshow(ir_image.img_data, cmap="gray")

        stacked_mask = np.zeros(shape=geometry_mask.pupil_mask.shape, dtype=int)
        stacked_mask[geometry_mask.eyeball_mask] = 1
        stacked_mask[geometry_mask.iris_mask] = 2
        stacked_mask[geometry_mask.pupil_mask] = 3

        axis.imshow(stacked_mask, alpha=0.5, cmap="jet")

        return fig, axis

    def plot_noise_mask(
        self,
        noise_mask: Union[iris_dc.NoiseMask, Dict[str, np.ndarray]],
        ir_image: Optional[Union[iris_dc.IRImage, Dict[str, Any]]] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualize an IRIS NoiseMask objet.

        Args:
            noise_mask (Union[iris_dc.NoiseMask, Dict[str, np.ndarray]]): Noise mask.
            ir_image (Optional[Union[iris_dc.IRImage, Dict[str, Any]]], optional): Optional IRImage to lay over in transparency. Defaults to None.
            ax (Optional[matplotlib.axes._axes.Axes], optional): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(noise_mask, dict):
            noise_mask = iris_dc.NoiseMask.deserialize(noise_mask)
        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)

        if ir_image is not None:
            fig, axis = self.plot_ir_image(ir_image, ax=axis)

        nm = noise_mask.mask.astype(np.float64)
        nm[nm == 0] = np.nan
        axis.imshow(nm, alpha=1, cmap="Reds", vmin=-1, vmax=0)

        return fig, axis

    def plot_geometry_polygons(
        self,
        geometry_polygons: Union[iris_dc.GeometryPolygons, Dict[str, np.ndarray]],
        ir_image: Optional[Union[iris_dc.IRImage, Dict[str, Any]]] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        scatter_kwargs: Optional[Dict[str, Any]] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualise an IRIS GeometryPolygons object.

        Args:
            geometry_polygons (Union[iris_dc.GeometryPolygons, Dict[str, np.ndarray]]): Geometry polygons to visualise
            ir_image (Optional[Union[iris_dc.IRImage, Dict[str, Any]]], optional): Optional IRImage to lay over in transparency. Defaults to None.
            plot_kwargs (Optional[Dict[str, Any]], optional): Kwargs of a plot function. Defaults to None.
            scatter_kwargs (Optional[Dict[str, Any]], optional): Kwargs of a scatter function. Defaults to None.
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(geometry_polygons, dict):
            geometry_polygons = iris_dc.GeometryPolygons.deserialize(geometry_polygons)
        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)

        if ir_image is not None:
            fig, axis = self.plot_ir_image(ir_image, ax=axis)

        if plot_kwargs is not None:
            axis.plot(*geometry_polygons.eyeball_array.T, **plot_kwargs)
            axis.plot(*geometry_polygons.iris_array.T, **plot_kwargs)
            axis.plot(*geometry_polygons.pupil_array.T, **plot_kwargs)
        elif scatter_kwargs is not None:
            axis.plot(*geometry_polygons.eyeball_array.T, **scatter_kwargs)
            axis.plot(*geometry_polygons.iris_array.T, **scatter_kwargs)
            axis.plot(*geometry_polygons.pupil_array.T, **scatter_kwargs)
        else:
            axis.plot(*geometry_polygons.eyeball_array.T)
            axis.plot(*geometry_polygons.iris_array.T)
            axis.plot(*geometry_polygons.pupil_array.T)

        return fig, axis

    def plot_eye_orientation(
        self,
        eye_orientation: Union[iris_dc.EyeOrientation, float],
        eye_centers: Union[iris_dc.EyeCenters, Dict[str, Tuple[float]]],
        ir_image: Optional[Union[iris_dc.IRImage, Dict[str, Any]]] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualise an IRIS EyeOrientation object. Require an EyeCenters.

        Args:
            eye_orientation (Union[iris_dc.EyeOrientation, float]): EyeOrientation to visualise
            eye_centers (Union[iris_dc.EyeCenters, Dict[str, Tuple[float]]]): EyeCenters, from which the origin of the displayed vector is inferred.
            ir_image (Optional[Union[iris_dc.IRImage, Dict[str, Any]]], optional): optional IRImage to lay over in transparency. Defaults to None.
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: output figure and axes
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(eye_orientation, float):
            eye_orientation = iris_dc.EyeOrientation.deserialize(eye_orientation)
        if isinstance(eye_centers, dict):
            eye_centers = iris_dc.EyeCenters.deserialize(eye_centers)
        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)

        if ir_image is not None:
            fig, axis = self.plot_ir_image(ir_image, ax=axis)

        axis.plot(
            [eye_centers.pupil_x, eye_centers.pupil_x + 200 * np.cos(eye_orientation.angle)],
            [eye_centers.pupil_y, eye_centers.pupil_y + 200 * np.sin(eye_orientation.angle)],
            label=f"eye orientation: {np.degrees(eye_orientation.angle):.3f} º",
        )
        axis.legend()

        return fig, axis

    def plot_eye_centers(
        self,
        eye_centers: Union[iris_dc.EyeCenters, Dict[str, Tuple[float]]],
        ir_image: Optional[Union[iris_dc.IRImage, Dict[str, Any]]] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualise an IRIS EyeCenters object.

        Args:
            eye_center (Union[iris_dc.EyeCenters, Dict[str, Tuple[float]]]): Eye centers to visualise
            ir_image (Optional[Union[iris_dc.IRImage, Dict[str, Any]]], optional): Optional IRImage to lay over in transparency. Defaults to None.
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(eye_centers, dict):
            eye_centers = iris_dc.EyeCenters.deserialize(eye_centers)
        if isinstance(ir_image, dict):
            ir_image = iris_dc.IRImage.deserialize(ir_image)

        if ir_image is not None:
            fig, axis = self.plot_ir_image(ir_image, ax=axis)

        axis.scatter([eye_centers.pupil_x], [eye_centers.pupil_y], label="pupil center", marker="+", s=100)
        axis.scatter([eye_centers.iris_x], [eye_centers.iris_y], label="iris center", marker="x", s=60)

        axis.legend()

        return fig, axis

    def plot_all_geometry(
        self,
        ir_image: Union[iris_dc.IRImage, Dict[str, Any]],
        geometry_polygons: Union[iris_dc.GeometryPolygons, Dict[str, np.ndarray]],
        eye_orientation: Union[iris_dc.EyeOrientation, float],
        eye_center: Union[iris_dc.EyeCenters, Dict[str, Tuple[float]]],
    ) -> Canvas:
        """Visualises all major geometry-related objects produced at an IR image level.

        This function displays the source IR Image, the produced GeometryPolygons, the eye orientation,
        and the pupil and iris centers.

        Args:
            ir_image (iris_dc.IRImage): source IR Image
            geometry_polygons (Union[iris_dc.GeometryPolygons, Dict[str, np.ndarray]]): geometry polygons from any stage of the pipeline
            eye_orientation (Union[iris_dc.EyeOrientation, float]): eye orientation to visualise
            eye_center (Union[iris_dc.EyeCenters, Dict[str, Tuple[float]]]): eye centers to visualise

        Returns:
            Canvas: Figure and axes.
        """
        fig, ax = self.plot_ir_image(ir_image=ir_image)
        fig, ax = self.plot_geometry_polygons(geometry_polygons, ax=ax)
        fig, ax = self.plot_eye_orientation(eye_orientation, eye_center, ax=ax)
        fig, ax = self.plot_eye_centers(eye_center, ax=ax)

        return fig, ax

    def plot_normalized_iris(
        self,
        normalized_iris: Union[iris_dc.NormalizedIris, Dict[str, np.ndarray]],
        plot_mask: bool = True,
        stretch_hist: bool = True,
        exposure_factor: float = 1.0,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualise an IRIS NormalizedIris object.

        Args:
            normalized_iris (Union[iris_dc.NormalizedIris, Dict[str, np.ndarray]]): Normalized iris image
            plot_mask (bool, optional): Wether to overlay the normalised mask in transparency. Defaults to True.
            stretch_hist (bool, optional): Wether to ignore masked out pixels in the image histogram. Useful for darker images. Defaults to True.
            exposure_factor (float, optional): Multiplicative factor to brighten the image. Defaults to 1.0.
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        canvas: Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes] = self._init_canvas(ax)
        fig, axis = canvas

        if isinstance(normalized_iris, dict):
            normalized_iris = iris_dc.NormalizedIris.deserialize(normalized_iris)

        axis.imshow(np.minimum(normalized_iris.normalized_image * exposure_factor, 255), cmap="gray")
        if stretch_hist:
            norm = normalized_iris.normalized_image * normalized_iris.normalized_mask
            norm = norm.astype(np.float32)
            norm[norm == 0] = np.nan
            axis.imshow(norm, cmap="gray")

        if plot_mask:
            nm = normalized_iris.normalized_mask.astype(np.float64)
            nm[nm == 1] = np.nan
            axis.imshow(nm, alpha=0.3, cmap="Reds", vmin=-1, vmax=3)

        return fig, axis

    def plot_iris_filter_response(
        self,
        iris_filter_response: Union[iris_dc.IrisFilterResponse, Dict[str, List[np.ndarray]]],
        space: Literal["cartesian", "polar"] = "cartesian",
        plot_mask: bool = True,
        mask_threshold: float = 0.9,
        vlim: float = 1e-3,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualise an IRIS IrisFilterResponse object.

        Args:
            iris_filter_response (Union[iris_dc.IrisFilterResponse, Dict[str, List[np.ndarray]]]): iris filter response to visualise.
            space (Literal["cartesian", "polar"], optional): Wether to plot the response in cartesian or polar coordinates. Defaults to cartesian.
            plot_mask (bool, optional): Wether to overlay the mask response in transparency. Defaults to True.
            mask_threshold (float, optional): Wether to overlay the mask in transparency. Defaults to 0.9.
            vlim (float, optional): The maximal value displayed in real, imaginary and amplitude graphs. Defaults to 1e-3
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        if isinstance(iris_filter_response, dict):
            iris_filter_response = iris_dc.IrisFilterResponse.deserialize(iris_filter_response)

        canvas: Tuple[matplotlib.figure.Figure, np.ndarray] = self._init_canvas(
            ax, subplot_size=(2 * len(iris_filter_response.iris_responses), 1)
        )
        fig, axs = canvas

        for i, (iris_response, mask_response) in enumerate(
            zip(iris_filter_response.iris_responses, iris_filter_response.mask_responses)
        ):
            ir = iris_response.copy()
            ir[mask_response <= mask_threshold] = np.nan
            mr = np.zeros_like(np.real(ir))
            mr[mask_response > mask_threshold] = np.nan

            if space == "cartesian":
                axs[2 * i].imshow(np.real(ir), cmap="seismic", vmin=-vlim, vmax=vlim)
                axs[2 * i + 1].imshow(np.imag(ir), cmap="seismic", vmin=-vlim, vmax=vlim)
                axs[2 * i].set_title(f"Wavelet {i}, Real part.")
                axs[2 * i + 1].set_title(f"Wavelet {i}, Imaginary part.")

            else:
                axs[2 * i].imshow(np.angle(ir), cmap=self.cyclic_cmap)
                axs[2 * i + 1].imshow(np.power(np.abs(ir), 2 / 3), cmap=self.linear_cmap, vmin=0, vmax=vlim ** (2 / 3))
                axs[2 * i].set_title(f"Wavelet {i}, Phase.")
                axs[2 * i + 1].set_title(f"Wavelet {i}, Amplitude.")

            if plot_mask:
                axs[2 * i].imshow(mr, cmap="gray")
                axs[2 * i + 1].imshow(mr, cmap="gray")

        return fig, axs

    def plot_iris_template(
        self,
        iris_template: Union[iris_dc.IrisTemplate, Dict[str, np.ndarray]],
        plot_mask: bool = True,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualise an IRIS IrisTemplate object.

        Args:
            iris_template (Union[iris_dc.IrisTemplate, Dict[str, np.ndarray]]): iris template to visualise
            plot_mask (bool, optional): Wether to overlay the mask in transparency. Defaults to True.
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        if isinstance(iris_template, dict):
            iris_template = self._deserialize_iris_template(iris_template)

        fig, axs = self._init_canvas(ax, subplot_size=(2 * len(iris_template.iris_codes), 1))

        for i, (iris_code, mask_code) in enumerate(zip(iris_template.iris_codes, iris_template.mask_codes)):
            axs[2 * i].imshow(iris_code[:, :, 0], cmap="gray")
            axs[2 * i + 1].imshow(iris_code[:, :, 1], cmap="gray")

            if plot_mask:
                nm = mask_code[:, :, 0].astype(np.float64)
                nm[nm == 1] = np.nan
                axs[2 * i].imshow(nm, alpha=0.8, cmap="Reds", vmin=-1, vmax=0)
                axs[2 * i + 1].imshow(nm, alpha=0.8, cmap="Reds", vmin=-1, vmax=0)

        return fig, axs

    def plot_iris_template_and_normalized_iris(
        self,
        iris_template: iris_dc.IrisTemplate,
        normalized_iris: Union[iris_dc.NormalizedIris, Dict[str, np.ndarray]],
        plot_mask: bool = True,
        linewidth: float = 0.5,
        fill_alpha: float = 0.05,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
    ) -> Canvas:
        """Visualises a normalised iris image and its associated iris template.

        Args:
            iris_template (iris_dc.IrisTemplate): iris template to visualise
            normalized_iris (Union[iris_dc.NormalizedIris, Dict[str, np.ndarray]]): normalised iris to visualise
            plot_mask (bool, optional): Wether to overlay the mask in transparency. Defaults to True.
            linewidth (float, optional): line width of the iris template. Defaults to 0.5.
            fill_alpha (float, optional): transparency of the overlaid iris template. Defaults to 0.05.
            ax (Optional[matplotlib.axes._axes.Axes]): ax to plot the figure at. Defaults to None.

        Returns:
            Canvas: Figure and axes.
        """
        if isinstance(iris_template, dict):
            iris_template = self._deserialize_iris_template(iris_template)
        if isinstance(normalized_iris, dict):
            normalized_iris = iris_dc.NormalizedIris.deserialize(normalized_iris)

        fig, axs = self._init_canvas(ax, subplot_size=(2 * len(iris_template.iris_codes), 1))

        for i, (iris_code, mask_code) in enumerate(zip(iris_template.iris_codes, iris_template.mask_codes)):
            for j in [0, 1]:
                _, axs[2 * i + j] = self.plot_normalized_iris(
                    normalized_iris, plot_mask=True, stretch_hist=True, ax=axs[2 * i + j]
                )

                template_resized = self._resize(
                    array=iris_code[:, :, j],
                    target_shape=normalized_iris.normalized_image.shape[::-1],
                )
                template_resized = (template_resized < 0.5).astype(np.uint8)
                axs[2 * i + j].imshow(template_resized, alpha=fill_alpha, cmap="Reds", vmin=0, vmax=1)

                contours, _ = cv2.findContours(template_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = [np.squeeze(contour) for contour in contours]
                for contour in contours:
                    axs[2 * i + j].plot(*contour.T, color="Blue", linewidth=linewidth)

                if plot_mask:
                    mask_resized = self._resize(
                        array=mask_code[:, :, 1],
                        target_shape=normalized_iris.normalized_image.shape[::-1],
                    )
                    mask_resized[mask_resized > 0.9] = np.nan
                    axs[2 * i + j].imshow(mask_resized, alpha=0.8, cmap="gray", vmin=0, vmax=1)

        return fig, axs

    def _init_canvas(
        self, ax: Optional[Union[matplotlib.axes._axes.Axes, np.ndarray]], subplot_size: Tuple[int, int] = (1, 1)
    ) -> Canvas:
        """Initialise figure and axes without ticks.

        Args:
            ax (Optional[Union[matplotlib.axes._axes.Axes, np.ndarray]]): Incoming axes.
            subplot_size (Tuple[int, int], optional): figure layout. Defaults to (1, 1).

        Returns:
            Canvas: Figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots(*subplot_size)
        else:
            fig = plt.gcf()

        if isinstance(ax, np.ndarray):
            for individual_ax in ax.flatten():
                individual_ax.set_xticks([])
                individual_ax.set_yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        return fig, ax

    def _resize(self, array: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Reshape an input array of size (N, M) into an array of shape `target_shape`.

        Args:
            array (np.ndarray): input array
            target_shape (tuple): output array shape

        Returns:
            np.ndarray: output array
        """
        array_resized = cv2.resize(array.astype(np.float64), dsize=target_shape, interpolation=cv2.INTER_CUBIC).clip(
            min=0, max=1
        )

        return array_resized

    def _make_linear_cmap(self) -> col.LinearSegmentedColormap:
        """Make linear color map.

        Returns:
            col.LinearSegmentedColormap: Color map.
        """
        white = "#ffffff"
        black = "#000000"
        blue1 = "#8888ff"
        blue2 = "#000088"
        linmap = col.LinearSegmentedColormap.from_list("linmap", [white, blue1, blue2, black], N=256, gamma=1)

        return linmap

    def _make_cyclic_cmap(self) -> col.LinearSegmentedColormap:
        """Make cyclic color map.

        Returns:
            col.LinearSegmentedColormap: Color map.
        """
        white = "#ffffff"
        black = "#000000"
        red = "#ff0000"
        blue = "#0000ff"
        anglemap = col.LinearSegmentedColormap.from_list("anglemap", [black, red, white, blue, black], N=256, gamma=1)

        return anglemap

    def _deserialize_iris_template(self, iris_template: Dict[str, np.ndarray]) -> iris_dc.IrisTemplate:
        """Decode and deserialize iris template.

        Args:
            iris_template (Dict[str, np.ndarray]): Serialized and iris template.

        Returns:
            iris_dc.IrisTemplate: Deserialized object.
        """
        decoded_iris = iris_template["iris_codes"]
        decoded_mask = iris_template["mask_codes"]

        return iris_dc.IrisTemplate(
            iris_codes=[decoded_iris[..., i] for i in range(decoded_iris.shape[2])],
            mask_codes=[decoded_mask[..., i] for i in range(decoded_iris.shape[2])],
        )

from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(
        self,
        model_path: str | Path,
        feature_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_dim = feature_dim
        self.device = device

        self._compiled_model = None
        self._output = None

    def _get_model(self):
        if self._compiled_model is None:
            core = ov.Core()
            self._compiled_model = core.compile_model(
                str(self.model_path),
                self.device.upper(),
                {"PERFORMANCE_HINT": "LATENCY"},
            )
            self._output = self._compiled_model.output(0)
        return self._compiled_model

    @staticmethod
    def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (128, 256),
            interpolation=cv2.INTER_LINEAR,
        )

        image = image_rgb.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        return image.transpose(2, 0, 1)

    def extract_features(
        self,
        bgr_images: np.ndarray | list[np.ndarray],
    ) -> torch.Tensor:
        """
        Extract person features from one BGR image or a list of BGR images.

        Args:
            bgr_images: A BGR image or a list of BGR images.

        Returns:
            Normalized features with shape ``(N, feature_dim)`` where ``N`` is the number of input images.
        """

        if isinstance(bgr_images, np.ndarray):
            bgr_images = [bgr_images]
        if not bgr_images:
            raise ValueError("No images provided for feature extraction.")

        batch = np.stack(
            [self._preprocess(image) for image in bgr_images],
            axis=0,
        )

        model = self._get_model()
        output = model(batch)[self._output]
        features = torch.from_numpy(
            np.asarray(output, dtype=np.float32).copy()
        )

        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Expected features with shape (N, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )

        return F.normalize(features, p=2, dim=1)

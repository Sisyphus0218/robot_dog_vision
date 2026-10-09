from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor as TorchReIDFeatureExtractor


class FeatureExtractor:
    def __init__(
        self,
        model_name: str,
        model_path: str | Path,
        feature_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.feature_dim = feature_dim
        self.device = device

        self._model: TorchReIDFeatureExtractor | None = None

    def _get_model(self) -> TorchReIDFeatureExtractor:
        if self._model is None:
            self._model = TorchReIDFeatureExtractor(
                model_name=self.model_name,
                model_path=str(self.model_path),
                device=self.device,
                verbose=False,
            )
        return self._model

    @torch.inference_mode()
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

        rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in bgr_images]
        features = self._get_model()(rgb_images)
        return F.normalize(features, p=2, dim=1)

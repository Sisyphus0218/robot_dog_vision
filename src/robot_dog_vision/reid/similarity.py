import torch

from .feature_memory import FeatureMemory


def calculate_similarity_scores(
    query_features: torch.Tensor,
    target_feature_memory: FeatureMemory,
    top_k: int = 3,
) -> torch.Tensor:
    """Calculate each query's similarity score against the target features."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target_features = target_feature_memory.all_features()
    if target_features.size(0) == 0:
        raise ValueError("Target feature memory is empty")

    # [num_queries, feature_dim] @ [num_gallery, feature_dim].T
    # -> [num_queries, num_gallery]
    #            target_feature1  target_feature2 ...
    # query1     similarity1_1    similarity1_2
    # query2     similarity2_1    similarity2_2
    # ...
    query_features = query_features.to(
        device=target_features.device,
        dtype=target_features.dtype,
    )
    similarity_matrix = query_features @ target_features.T

    # Average each query's top-k similarity scores.
    num_neighbors = min(top_k, target_features.shape[0])
    similarity_scores = similarity_matrix.topk(
        k=num_neighbors,
        dim=1,
    ).values.mean(dim=1)

    return similarity_scores.detach()

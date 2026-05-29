from typing import Final
from src.features.feature_metadata import FeatureMetadata, FeatureKind, FeatureGroup, SITUNES_FEATURES_METADATA

class FeatureRegistry:
    """
    Provide structured access to FeatureMetadata objects.
    Convert a flat feature list into key:value structure to avoid manual searching.
    """
    def __init__(self, feature_list: list[FeatureMetadata]):
        self._features_map: dict[str, FeatureMetadata] = {
            feature.name: feature
            for feature in feature_list
        }

    def get(self, feature_name: str) -> FeatureMetadata:
        try:
            return self._features_map[feature_name]
        except KeyError:
            raise KeyError(f"Feature {feature_name} not found in registry.")
        
    
    def get_by_group(self, feature_group: FeatureGroup) -> list[FeatureMetadata]:
        return [
            feature
            for feature in self._features_map.values()
            if feature.group == feature_group
        ]
    
    def get_by_kind(self, feature_kind: FeatureKind) -> list[FeatureMetadata]:
        return [
            feature
            for feature in self._features_map.values()
            if feature.kind == feature_kind
        ]
        
    @property
    def all_features(self) -> list[FeatureMetadata]:
        """Return the original flat list of features."""
        return list(self._features_map.values())
    

FEATURE_REGISTRY: Final[FeatureRegistry] = FeatureRegistry(SITUNES_FEATURES_METADATA)
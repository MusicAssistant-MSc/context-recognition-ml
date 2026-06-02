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

    def __contains__(self, feature_name: str) -> bool:
        return feature_name in self._features_map

    def __getitem__(self, feature_name: str) -> FeatureMetadata:
        return self.get(feature_name)
    
    def __iter__(self):
        return iter(self._features_map.values())
    
    def __len__(self):
        return len(self._features_map)

    def get(self, feature_name: str) -> FeatureMetadata:
        try:
            return self._features_map[feature_name]
        except KeyError:
            raise KeyError(f"Feature {feature_name} not found in registry.")
        
    def get_features(self, feature_names: list[str]) -> list[FeatureMetadata]:
        return [self.get(feature_name) for feature_name in feature_names]
    
    def get_names(self, features: list[FeatureMetadata]) -> list[str]:
        return [feature.name for feature in features]
        
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
    
    def filter(
        self,
        group: FeatureGroup | None = None,
        kind: FeatureKind | None = None,
        source: list[FeatureMetadata] | None = None
    ) -> list[FeatureMetadata]:
        """Universal method to filter features by group, kind, or source.

        Args:
            group (FeatureGroup | None, optional): Filter by feature group. Defaults to None.
            kind (FeatureKind | None, optional): Filter by feature kind. Defaults to None.
            source (list[FeatureMetadata] | None, optional): Filter by source features instead of whole registry. Defaults to None.

        Returns:
            list[FeatureMetadata]: List of filtered features.
        """
        filtered_features = source if source is not None else self.all_features
        if group:
            filtered_features = [feature for feature in filtered_features if feature.group == group]
        if kind:
            filtered_features = [feature for feature in filtered_features if feature.kind == kind]
        return filtered_features
        
    @property
    def all_features(self) -> list[FeatureMetadata]:
        """Return the original flat list of features."""
        return list(self._features_map.values())
    

FEATURE_REGISTRY: Final[FeatureRegistry] = FeatureRegistry(SITUNES_FEATURES_METADATA)
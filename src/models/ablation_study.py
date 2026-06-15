import logging
import pandas as pd
from itertools import combinations
from typing import Any

from src.models.const import (
    ModelAlgorithm, PreprocessingStrategy, ALGORITHM_PRECEPTION_MAP, ExperimentResult
)
from src.features.const import BioFeatureNames, MotionFeatureNames, ContextFeatureNames, MusicFeatureNames, EmotionFeatureNames
from src.models.training import trainer
from src.models import persistence

logger = logging.getLogger(__name__)


class AblationOrchestrator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        # Base mandatory features
        self.music_features = self._get_feature_list(MusicFeatureNames)
        self.emotion_features = self._get_feature_list(EmotionFeatureNames)
        
        # Subsets for power set
        self.subsets = {
            "bio": self._get_feature_list(BioFeatureNames),
            "motion": self._get_feature_list(MotionFeatureNames),
            "context": self._get_feature_list(ContextFeatureNames),
        }

    def _get_feature_list(self, feature_class: Any) -> list[str]:
        return [v for k, v in vars(feature_class).items() if not k.startswith("__")]
    
    def _generate_variants(self) -> dict[str, list[str]]:
        """Generate all possible combinations of feature subsets."""
        variants = {}
        keys = list(self.subsets.keys())
        
        # Generate combinations (1 to 3 subsets)
        for r in range(1, len(keys) + 1):
            for combo in combinations(keys, r):
                variant_name = "+".join(combo)
                # Always include music features
                feature_list = self.music_features.copy()
                # Add features from subsets combination
                for key in combo:
                    feature_list.extend(self.subsets[key])
                
                # Base variant (without emotions)
                variants[f"{variant_name}"] = feature_list
                
                # Variant with emotions
                variants[f"{variant_name}+emo"] = feature_list + self.emotion_features
                
        return variants
    
    def run_full_study(
        self, 
        algorithms: list[ModelAlgorithm], 
        preception_map: dict[ModelAlgorithm, PreprocessingStrategy] = ALGORITHM_PRECEPTION_MAP
    ) -> None:
        """Run the entire ablation study for a given list of algorithms.
        
        Args:
            algorithms (list[ModelAlgorithm]): List of algorithms to run the study for.
            preception_map (dict[ModelAlgorithm, PreprocessingStrategy], optional): Mapping of algorithms to preprocessing strategies. Defaults to ALGORITHM_PRECEPTION_MAP.

        Returns:
            None
        """
        variants = self._generate_variants()
        for algorithm in algorithms:
            preprocessing_strategy = preception_map[algorithm]
            for variant_name, feature_names in variants.items():
                logger.info("="*50)
                logger.info(f"Running study for algorithm: {algorithm} and variant: {variant_name}")

                try:
                    result = trainer.train_and_evaluate_model(
                        algorithm=algorithm,
                        preprocessing_strategy=preprocessing_strategy,
                        data=self.data,
                        feature_names=feature_names,
                        variant_name=variant_name
                    )
                    # Persist results
                    persistence.save_experiment_results(
                        experiment_result=result,
                        experiment_name=f"{algorithm.value}_{variant_name}"
                    )
                except Exception as e:
                    logger.error(f"Failed experiment {algorithm.value} {variant_name}: {e}")
"""Конфигурационные классы и утилиты для searchdet_pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import yaml
from .models import DetectorConfig, BatchProcessingConfig, MaskBackend, BackboneType


class ConfigManager:
    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> DetectorConfig:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {config_path.suffix}")
        
        return ConfigManager._dict_to_config(data)
    
    @staticmethod
    def save_to_file(config: DetectorConfig, config_path: Union[str, Path]) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {config_path.suffix}")
    
    @staticmethod
    def _config_to_dict(config: DetectorConfig) -> Dict[str, Any]:
        data = asdict(config)
        
        if isinstance(config.mask_backend, MaskBackend):
            data['mask_backend'] = config.mask_backend.value
        if isinstance(config.backbone, BackboneType):
            data['backbone'] = config.backbone.value
            
        return data
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> DetectorConfig:
        if 'mask_backend' in data and isinstance(data['mask_backend'], str):
            data['mask_backend'] = MaskBackend(data['mask_backend'])
        if 'backbone' in data and isinstance(data['backbone'], str):
            data['backbone'] = BackboneType(data['backbone'])
            
        return DetectorConfig(**data)
    
    @staticmethod
    def create_default_config() -> DetectorConfig:
        return DetectorConfig()
    
    @staticmethod
    def merge_configs(base_config: DetectorConfig, override_config: Dict[str, Any]) -> DetectorConfig:
        base_dict = ConfigManager._config_to_dict(base_config)
        base_dict.update(override_config)
        return ConfigManager._dict_to_config(base_dict)


@dataclass
class CLIConfig:
    verbose: bool = False
    quiet: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    config_file: Optional[str] = None
    save_config: Optional[str] = None
    
    def __post_init__(self):
        if self.verbose and self.quiet:
            raise ValueError("Нельзя одновременно использовать verbose и quiet режимы")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Неподдерживаемый уровень логирования: {self.log_level}")


class ConfigValidator:
    @staticmethod
    def validate_detector_config(config: DetectorConfig) -> None:
        errors = []
        
        model_paths = {
            'sam_model_path': config.sam_model_path,
            'sam_encoder_path': config.sam_encoder_path,
            'sam2_weights_path': config.sam2_weights_path,
            'fastsam_model_path': config.fastsam_model_path,
            'dinov3_checkpoint_path': config.dinov3_checkpoint_path
        }
        
        for path_name, path_value in model_paths.items():
            if path_value and not Path(path_value).exists():
                errors.append(f"Файл модели не найден: {path_name} = {path_value}")
        
        if config.min_confidence < 0 or config.min_confidence > 1:
            errors.append(f"min_confidence должен быть в диапазоне [0, 1]: {config.min_confidence}")
        
        if config.max_masks <= 0:
            errors.append(f"max_masks должен быть положительным: {config.max_masks}")
        
        if config.min_area_fraction < 0 or config.min_area_fraction > 1:
            errors.append(f"min_area_fraction должен быть в диапазоне [0, 1]: {config.min_area_fraction}")
        
        if config.max_area_fraction < 0 or config.max_area_fraction > 1:
            errors.append(f"max_area_fraction должен быть в диапазоне [0, 1]: {config.max_area_fraction}")
        
        if config.min_area_fraction >= config.max_area_fraction:
            errors.append("min_area_fraction должен быть меньше max_area_fraction")
        
        if config.score_confidence < 0 or config.score_confidence > 1:
            errors.append(f"score_confidence должен быть в диапазоне [0, 1]: {config.score_confidence}")
        
        if config.decision_threshold < 0 or config.decision_threshold > 1:
            errors.append(f"decision_threshold должен быть в диапазоне [0, 1]: {config.decision_threshold}")
        
        valid_aggregations = ["mean", "max", "median"]
        if config.positive_aggregation not in valid_aggregations:
            errors.append(f"positive_aggregation должен быть одним из {valid_aggregations}: {config.positive_aggregation}")
        
        valid_pooling = ["cls", "mean", "max"]
        if config.vit_pooling not in valid_pooling:
            errors.append(f"vit_pooling должен быть одним из {valid_pooling}: {config.vit_pooling}")
        
        if errors:
            raise ValueError("Ошибки валидации конфигурации:\n" + "\n".join(f"- {error}" for error in errors))
    
    @staticmethod
    def validate_batch_config(config: BatchProcessingConfig) -> None:
        errors = []
        
        if not config.input_directory.exists():
            errors.append(f"Входная директория не существует: {config.input_directory}")
        
        if not config.input_directory.is_dir():
            errors.append(f"Входной путь не является директорией: {config.input_directory}")
        
        if config.positive_examples_directory and not config.positive_examples_directory.exists():
            errors.append(f"Директория положительных примеров не существует: {config.positive_examples_directory}")
        
        if config.negative_examples_directory and not config.negative_examples_directory.exists():
            errors.append(f"Директория отрицательных примеров не существует: {config.negative_examples_directory}")
        
        if config.max_workers < 1:
            errors.append(f"max_workers должен быть >= 1: {config.max_workers}")
        
        if not config.file_extensions:
            errors.append("Список расширений файлов не может быть пустым")
        
        if errors:
            raise ValueError("Ошибки валидации конфигурации пакетной обработки:\n" + "\n".join(f"- {error}" for error in errors))


PRESET_CONFIGS = {
    "fast": DetectorConfig(
        mask_backend=MaskBackend.FASTSAM,
        backbone=BackboneType.DINOV2_S,
        half_precision=True,
        max_masks=500,
        fastsam_image_size=640,
        min_confidence=0.3
    ),
    
    "balanced": DetectorConfig(
        mask_backend=MaskBackend.FASTSAM,
        backbone=BackboneType.DINOV2_B,
        half_precision=False,
        max_masks=1000,
        fastsam_image_size=1024,
        min_confidence=0.5
    ),
    
    "quality": DetectorConfig(
        mask_backend=MaskBackend.SAM,
        backbone=BackboneType.DINOV3_VITL16,
        half_precision=False,
        max_masks=2000,
        sam_long_side=1024,
        min_confidence=0.7,
        smart_rectangle_filter=True
    ),
    
    "defect_detection": DetectorConfig(
        mask_backend=MaskBackend.SAM_HQ,
        backbone=BackboneType.DINOV3_VITH14,
        defect_mode=True,
        min_confidence=0.3,
        min_area_fraction=0.0001,
        max_area_fraction=0.5,
        smart_rectangle_filter=False
    )
}


def get_preset_config(preset_name: str) -> DetectorConfig:
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Неизвестный пресет: {preset_name}. Доступные: {available}")
    
    return PRESET_CONFIGS[preset_name]


def list_preset_configs() -> Dict[str, str]:
    descriptions = {
        "fast": "Быстрая обработка с пониженным качеством",
        "balanced": "Сбалансированное соотношение скорости и качества",
        "quality": "Высокое качество детекции (медленнее)",
        "defect_detection": "Специализированная конфигурация для поиска дефектов"
    }
    return descriptions
"""Утилиты для валидации входных данных и обработки ошибок."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch


class ValidationError(Exception):
    pass


class ImageValidator:
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
    MIN_IMAGE_DIMENSION = 32
    MAX_IMAGE_DIMENSION = 8192
    
    @classmethod
    def validate_image_path(cls, image_path: Union[str, Path]) -> Path:
        """Валидирует путь к изображению."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise ValidationError(f"Файл изображения не найден: {image_path}")
        
        if not image_path.is_file():
            raise ValidationError(f"Путь не является файлом: {image_path}")
        
        if image_path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            supported = ', '.join(cls.SUPPORTED_FORMATS)
            raise ValidationError(
                f"Неподдерживаемый формат изображения: {image_path.suffix}. "
                f"Поддерживаемые форматы: {supported}"
            )
        

        file_size = image_path.stat().st_size
        if file_size > cls.MAX_IMAGE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_mb = cls.MAX_IMAGE_SIZE / (1024 * 1024)
            raise ValidationError(
                f"Размер файла слишком большой: {size_mb:.1f}MB. "
                f"Максимальный размер: {max_mb}MB"
            )
        
        return image_path
    
    @classmethod
    def validate_image_content(cls, image_path: Union[str, Path]) -> Tuple[int, int, str]:
        image_path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                

                if width < cls.MIN_IMAGE_DIMENSION or height < cls.MIN_IMAGE_DIMENSION:
                    raise ValidationError(
                        f"Изображение слишком маленькое: {width}x{height}. "
                        f"Минимальный размер: {cls.MIN_IMAGE_DIMENSION}x{cls.MIN_IMAGE_DIMENSION}"
                    )
                
                if width > cls.MAX_IMAGE_DIMENSION or height > cls.MAX_IMAGE_DIMENSION:
                    raise ValidationError(
                        f"Изображение слишком большое: {width}x{height}. "
                        f"Максимальный размер: {cls.MAX_IMAGE_DIMENSION}x{cls.MAX_IMAGE_DIMENSION}"
                    )
                

                if mode not in ['RGB', 'RGBA', 'L', 'P']:
                    raise ValidationError(f"Неподдерживаемый цветовой режим: {mode}")
                
                return width, height, mode
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Ошибка при чтении изображения {image_path}: {e}")
    
    @classmethod
    def validate_image_array(cls, image_array: np.ndarray) -> None:
        """Валидирует массив изображения."""
        if not isinstance(image_array, np.ndarray):
            raise ValidationError(f"Ожидался numpy.ndarray, получен {type(image_array)}")
        
        if image_array.ndim not in [2, 3]:
            raise ValidationError(f"Изображение должно быть 2D или 3D массивом, получен {image_array.ndim}D")
        
        if image_array.ndim == 3 and image_array.shape[2] not in [1, 3, 4]:
            raise ValidationError(f"Количество каналов должно быть 1, 3 или 4, получено {image_array.shape[2]}")
        
        height, width = image_array.shape[:2]
        if height < cls.MIN_IMAGE_DIMENSION or width < cls.MIN_IMAGE_DIMENSION:
            raise ValidationError(
                f"Изображение слишком маленькое: {width}x{height}. "
                f"Минимальный размер: {cls.MIN_IMAGE_DIMENSION}x{cls.MIN_IMAGE_DIMENSION}"
            )


class DirectoryValidator:
    """Валидатор для директорий."""
    
    @staticmethod
    def validate_input_directory(directory_path: Union[str, Path], 
                               required: bool = True) -> Optional[Path]:
        """Валидирует входную директорию."""
        if directory_path is None:
            if required:
                raise ValidationError("Путь к директории не может быть None")
            return None
        
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            if required:
                raise ValidationError(f"Директория не найдена: {directory_path}")
            return None
        
        if not directory_path.is_dir():
            raise ValidationError(f"Путь не является директорией: {directory_path}")
        

        if not os.access(directory_path, os.R_OK):
            raise ValidationError(f"Нет прав на чтение директории: {directory_path}")
        
        return directory_path
    
    @staticmethod
    def validate_output_directory(directory_path: Union[str, Path], 
                                create_if_missing: bool = True) -> Path:
        """Валидирует выходную директорию."""
        directory_path = Path(directory_path)
        
        if directory_path.exists():
            if not directory_path.is_dir():
                raise ValidationError(f"Путь существует, но не является директорией: {directory_path}")
            

            if not os.access(directory_path, os.W_OK):
                raise ValidationError(f"Нет прав на запись в директорию: {directory_path}")
        else:
            if create_if_missing:
                try:
                    directory_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(f"Не удалось создать директорию {directory_path}: {e}")
            else:
                raise ValidationError(f"Выходная директория не существует: {directory_path}")
        
        return directory_path
    
    @staticmethod
    def find_images_in_directory(directory_path: Path, 
                               recursive: bool = False,
                               extensions: Optional[List[str]] = None) -> List[Path]:
        """Находит все изображения в директории."""
        if extensions is None:
            extensions = list(ImageValidator.SUPPORTED_FORMATS)
        
        extensions = [ext.lower() for ext in extensions]
        
        images = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    ImageValidator.validate_image_path(file_path)
                    images.append(file_path)
                except ValidationError as e:
                    logging.warning(f"Пропуск файла {file_path}: {e}")
        
        return sorted(images)


class ModelValidator:
    """Валидатор для моделей и их параметров."""
    
    @staticmethod
    def validate_device(device: str) -> str:
        """Валидирует и нормализует устройство."""
        device = device.lower().strip()
        
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                logging.warning("CUDA недоступна, используется CPU")
                return "cpu"
            return "cuda"
        elif device == "cpu":
            return "cpu"
        else:
            raise ValidationError(f"Неподдерживаемое устройство: {device}. Используйте 'auto', 'cuda' или 'cpu'")
    
    @staticmethod
    def validate_model_path(model_path: Union[str, Path], required: bool = True) -> Optional[Path]:
        """Валидирует путь к модели."""
        if model_path is None:
            if required:
                raise ValidationError("Путь к модели не может быть None")
            return None
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ValidationError(f"Файл модели не найден: {model_path}")
        
        if not model_path.is_file():
            raise ValidationError(f"Путь к модели не является файлом: {model_path}")
        

        valid_extensions = {'.pth', '.pt', '.ckpt', '.safetensors', '.bin'}
        if model_path.suffix.lower() not in valid_extensions:
            extensions_str = ', '.join(valid_extensions)
            raise ValidationError(
                f"Неподдерживаемое расширение файла модели: {model_path.suffix}. "
                f"Поддерживаемые: {extensions_str}"
            )
        
        return model_path
    
    @staticmethod
    def validate_confidence_threshold(confidence: float, name: str = "confidence") -> None:
        """Валидирует порог уверенности."""
        if not isinstance(confidence, (int, float)):
            raise ValidationError(f"{name} должен быть числом, получен {type(confidence)}")
        
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(f"{name} должен быть в диапазоне [0, 1], получен {confidence}")
    
    @staticmethod
    def validate_positive_integer(value: int, name: str, min_value: int = 1) -> None:
        """Валидирует положительное целое число."""
        if not isinstance(value, int):
            raise ValidationError(f"{name} должен быть целым числом, получен {type(value)}")
        
        if value < min_value:
            raise ValidationError(f"{name} должен быть >= {min_value}, получен {value}")


class MaskValidator:
    """Валидатор для масок."""
    
    @staticmethod
    def validate_mask_array(mask: np.ndarray) -> None:
        """Валидирует массив маски."""
        if not isinstance(mask, np.ndarray):
            raise ValidationError(f"Маска должна быть numpy.ndarray, получен {type(mask)}")
        
        if mask.ndim != 2:
            raise ValidationError(f"Маска должна быть 2D массивом, получен {mask.ndim}D")
        
        if mask.dtype not in [bool, np.bool_, np.uint8]:
            raise ValidationError(f"Маска должна быть bool или uint8, получен {mask.dtype}")
        

        if mask.dtype == bool:
            if not mask.any():
                raise ValidationError("Маска не может быть полностью пустой")
        else:  # uint8
            if not (mask > 0).any():
                raise ValidationError("Маска не может быть полностью пустой")
    
    @staticmethod
    def validate_mask_dimensions(mask: np.ndarray, image_shape: Tuple[int, int]) -> None:
        """Валидирует размеры маски относительно изображения."""
        if mask.shape != image_shape[:2]:
            raise ValidationError(
                f"Размеры маски {mask.shape} не соответствуют размерам изображения {image_shape[:2]}"
            )


class EmbeddingValidator:
    """Валидатор для эмбеддингов."""
    
    @staticmethod
    def validate_embedding_array(embedding: np.ndarray) -> None:
        """Валидирует массив эмбеддинга."""
        if not isinstance(embedding, np.ndarray):
            raise ValidationError(f"Эмбеддинг должен быть numpy.ndarray, получен {type(embedding)}")
        
        if embedding.ndim != 1:
            raise ValidationError(f"Эмбеддинг должен быть 1D массивом, получен {embedding.ndim}D")
        
        if embedding.size == 0:
            raise ValidationError("Эмбеддинг не может быть пустым")
        
        if not np.isfinite(embedding).all():
            raise ValidationError("Эмбеддинг содержит NaN или бесконечные значения")
    
    @staticmethod
    def validate_embedding_dimensions(embeddings: List[np.ndarray]) -> None:
        """Валидирует совместимость размерностей эмбеддингов."""
        if not embeddings:
            return
        
        first_dim = embeddings[0].shape[0]
        for i, emb in enumerate(embeddings[1:], 1):
            if emb.shape[0] != first_dim:
                raise ValidationError(
                    f"Несовместимые размерности эмбеддингов: {first_dim} vs {emb.shape[0]} (индекс {i})"
                )


def validate_processing_pipeline_inputs(image_path: Union[str, Path],
                                      positive_dir: Optional[Union[str, Path]] = None,
                                      negative_dir: Optional[Union[str, Path]] = None,
                                      output_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Optional[Path], Optional[Path], Path]:
    """Комплексная валидация входных данных для пайплайна обработки."""
    
    # Валидация основного изображения
    image_path = ImageValidator.validate_image_path(image_path)
    ImageValidator.validate_image_content(image_path)
    
    # Валидация директорий с примерами
    if positive_dir is not None:
        positive_dir = DirectoryValidator.validate_input_directory(positive_dir, required=False)
    if negative_dir is not None:
        negative_dir = DirectoryValidator.validate_input_directory(negative_dir, required=False)
    
    # # Проверка наличия хотя бы одной директории с примерами
    # if positive_dir is None and negative_dir is None:
    #     raise ValidationError("Должна быть указана хотя бы одна директория с примерами (positive или negative)")
    
    # Валидация выходной директории
    if output_dir is None:
        output_dir = Path("output")
    output_dir = DirectoryValidator.validate_output_directory(output_dir, create_if_missing=True)
    
    return image_path, positive_dir, negative_dir, output_dir
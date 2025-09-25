from dataclasses import dataclass, field
from typing import List
import numpy as np
import uuid

#Этот класс служит для описание каждого изображения 
@dataclass
class COCOAnnotation:
    #Супер базовые поля,которые должны быть
    img:np.ndarray
    mask:np.ndarray # Выполняет роль segmentation в COCO standara format
    label: int | str
    image_size: tuple[int, int]  # width, height (aod) change name
    
    width: int
    height: int

    area: float
    file_name: str
    bbox: List[float] = field(default_factory=list)
    score: float | None = None
    confidence: float | None = None
    
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    #Необходимость этих полей спорна,хотя в COCO Format они есть,куча этих полей вроде как можно откинуть :
    # license: Optional[int] = None
    # flickr_url: Optional[str] = None
    # coco_url: Optional[str] = None
    # date_captured: Optional[str] = None
    # year: Optional[int]
    # version: Optional[str]
    # description: Optional[str]
    # contributor: Optional[str]
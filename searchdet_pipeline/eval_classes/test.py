from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional
import xml.etree.ElementTree as ET

from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset


ImageDir = Optional[Path]


def _collect_xmls(base_dir: Path) -> List[Path]:
    # Рекурсивный поиск XML с регистронезависимой фильтрацией
    return sorted(p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")


def _find_image_dirs(base_dir: Path) -> List[Path]:
    candidates = [
        base_dir / "images",
        base_dir / "Images",
        base_dir / "image",
        base_dir / "Image",
        base_dir / "JPEGImages",
        base_dir / "jpegimages",
        base_dir / "JPEGIMAGES",
        base_dir,
    ]
    return [d for d in candidates if d.exists() and d.is_dir()]


def _count_images_in_dir(img_dir: Path) -> int:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts)


def _extract_filename_from_xml(xml_path: Path) -> str:
    try:
        root = ET.parse(xml_path).getroot()
        fn = root.findtext("filename", default="")
        return (fn or "").strip()
    except Exception:
        return ""


def _resolve_img_path(file_name: str, img_dirs: List[Path]) -> Optional[Path]:
    if not file_name:
        return None
    fn_path = Path(file_name)
    if fn_path.is_absolute():
        return fn_path if fn_path.exists() else None
    for d in img_dirs:
        cand = d / file_name
        if cand.exists():
            return cand
        # fallback на родителя папки изображений (учёт вложенных путей из xml)
        alt = d.parent / file_name
        if alt.exists():
            return alt
    return None


def main() -> int:
    # Поддержка: python test.py [ПУТЬ_К_ДАТАСЕТУ] [--ann-dir ПУТЬ_К_АННОТАЦИЯМ] [--img-dir ПУТЬ_К_ИЗОБРАЖЕНИЯМ]
    args = sys.argv[1:]
    dataset_dir = Path("archive")
    ann_dir: Optional[Path] = None
    img_dir: Optional[Path] = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--dataset", "--root", "--data"):
            if i + 1 < len(args):
                dataset_dir = Path(args[i + 1])
                i += 2
                continue
        if arg in ("--ann", "--ann-dir", "--annotations"):
            if i + 1 < len(args):
                ann_dir = Path(args[i + 1])
                i += 2
                continue
        if arg in ("--img", "--img-dir", "--images"):
            if i + 1 < len(args):
                img_dir = Path(args[i + 1])
                i += 2
                continue
        # Первый позиционный аргумент — корень датасета, если не указан через флаг
        if not arg.startswith("-") and str(dataset_dir) == "archive":
            dataset_dir = Path(arg)
            i += 1
            continue
        i += 1

    print(f"[i] Рабочая папка: {Path.cwd().resolve()}")
    print(f"[i] Указанный путь к датасету: {dataset_dir} -> {dataset_dir.resolve()}")
    if ann_dir is not None:
        print(f"[i] Папка аннотаций (--ann-dir): {ann_dir} -> {ann_dir.resolve()}")
    if img_dir is not None:
        print(f"[i] Папка изображений (--img-dir): {img_dir} -> {img_dir.resolve()}")

    if not dataset_dir.exists():
        print(f"Папка с датасетом не найдена: {dataset_dir.resolve()}")
        return 1

    # Диагностика структуры до загрузки датасета
    ann_scan_dir = ann_dir if ann_dir is not None else dataset_dir
    xml_files = _collect_xmls(ann_scan_dir)
    img_dirs = [img_dir] if img_dir is not None else _find_image_dirs(dataset_dir)

    print("\nДиагностика структуры датасета (до парсинга):")
    print(f"  • Найдено XML-аннотаций: {len(xml_files)}")
    if xml_files:
        print(f"    Примеры XML: {[str(p.relative_to(ann_scan_dir)) for p in xml_files[:3]]}")
    else:
        print("    Внимание: XML не найдены. Проверьте, что аннотации лежат в annotations/ или рядом с изображениями." )

    if img_dirs:
        base_for_rel = img_dir if img_dir is not None else dataset_dir
        rel_list = []
        for d in img_dirs:
            try:
                rel_list.append(str(d.relative_to(base_for_rel)))
            except Exception:
                rel_list.append(str(d))
        print(f"  • Кандидат(ы) папок с изображениями: {rel_list}")
        for d in img_dirs:
            try:
                cnt = _count_images_in_dir(d)
            except Exception:
                cnt = 0
            print(f"    - {d.name}: {cnt} файлов-изображений")
    else:
        print("  • Папки с изображениями не найдены среди стандартных имён (images, JPEGImages и т.п.)")

    # Проверим соответствие filename из первых XML фактическим файлам
    if xml_files and img_dirs:
        print("  • Проверка соответствия filename -> изображение (первые 5):")
        for xp in xml_files[:5]:
            fn = _extract_filename_from_xml(xp)
            img_p = _resolve_img_path(fn, img_dirs)
            status = "OK" if img_p and img_p.exists() else "NOT FOUND"
            print(f"    - xml={xp.name}, filename='{fn}' -> {status}{' (' + str(img_p) + ')' if img_p else ''}")

    # Загрузка датасета
    try:
        dataset = ArchiveVOCDataset.from_path(dataset_dir, ann_dir=ann_dir, img_dir=img_dir)
    except Exception as e:
        print(f"\nОшибка при загрузке датасета из {dataset_dir}: {e}")
        return 2

    dm = dataset.data
    anns = list(dm.data_points)
    total_annotations = len(anns)
    file_names = sorted({getattr(a, "file_name", None) for a in anns if getattr(a, "file_name", None)})
    total_images = len(file_names)
    categories = sorted({str(getattr(a, "label", "")) for a in anns if getattr(a, "label", None) is not None})

    print("\nИнформация о датасете")
    print(f"Имя датасета: {dataset.name}")
    print(f"Корневая папка: {dataset_dir.resolve()}")
    print(f"Всего изображений: {total_images}")
    print(f"Всего аннотаций: {total_annotations}")
    print(f"Категории ({len(categories)}): {categories}")

    per_class: Dict[Any, int] = {}
    for a in anns:
        label = getattr(a, "label", None)
        if label is None:
            continue
        per_class[label] = per_class.get(label, 0) + 1

    if per_class:
        print("Аннотаций по категориям:")
        for label in sorted(per_class, key=lambda x: str(x)):
            print(f"  - {label}: {per_class[label]}")

    sample_n = min(5, total_annotations)
    if sample_n > 0:
        print(f"\nПервые {sample_n} аннотаций:")
        for a in anns[:sample_n]:
            file_name = getattr(a, "file_name", None)
            label = getattr(a, "label", None)
            bbox = getattr(a, "bbox", None)
            width = getattr(a, "width", None)
            height = getattr(a, "height", None)
            print(f"- file: {file_name} | label: {label} | bbox: {bbox} | size: {width}x{height}")

    try:
        print("\nМетаданные датасета:")
        print(dm.meta)
    except Exception:
        pass

    # Финальная подсказка, если всё по нулям
    if total_annotations == 0:
        print("\n[!] Похоже, аннотации не были распознаны. Проверьте:")
        print("   1) Путь к датасету (запустите: python test.py /абсолютный/путь/к/archive)")
        print("   2) Наличие XML (см. диагностику выше) или укажите: --ann-dir /путь/к/annotations")
        print("   3) Содержимое тега <filename> в XML — совпадает ли с реальными именами файлов?")
        print("   4) Что изображения действительно лежат в одной из папок: images, JPEGImages, image (или укажите: --img-dir /путь/к/images)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
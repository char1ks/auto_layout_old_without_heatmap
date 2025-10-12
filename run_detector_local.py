from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
from PIL import Image
from flashbone.app import init_detector  


def load_positive_images(pos_dir: Path) -> Dict[str, List[Image.Image]]:
    pos_by_class: Dict[str, List[Image.Image]] = {}
    for class_dir in sorted(p for p in pos_dir.iterdir() if p.is_dir()):
        class_id = class_dir.name
        imgs: List[Image.Image] = []
        for img_path in sorted(class_dir.glob("*")):
            try:
                imgs.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"[WARN] Не удалось открыть {img_path}: {e}")
        if imgs:
            pos_by_class[class_id] = imgs
    return pos_by_class


def load_images_from_dir(img_dir: Path) -> List[Image.Image]:
    images: List[Image.Image] = []
    for img_path in sorted(img_dir.glob("*")):
        try:
            images.append(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"[WARN] Не удалось открыть {img_path}: {e}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Local detector runner")
    parser.add_argument("--pos_dir", type=Path, required=True, help="Директория с положительными примерами (по подпапкам классов)")
    parser.add_argument("--neg_dir", type=Path, required=False, help="Директория с отрицательными примерами")
    parser.add_argument("--search_dir", type=Path, required=True, help="Директория с изображениями для поиска/детекции")
    parser.add_argument("--heatmap_thresh", type=float, default=None, help="Порог для heatmap (None = использовать дефолт)" )
    parser.add_argument("--class_thresh", type=float, default=0.5, help="Порог для классификатора")
    args = parser.parse_args()
    print("[INFO] Инициализация детектора …")
    detector = init_detector()
    print("[INFO] Загрузка положительных изображений …")
    pos_by_class = load_positive_images(args.pos_dir)
    neg_imgs: List[Image.Image] = []
    if args.neg_dir and args.neg_dir.exists():
        print("[INFO] Загрузка отрицательных изображений …")
        neg_imgs = load_images_from_dir(args.neg_dir)

    print(
        f"[INFO] Обучение детектора: {sum(len(v) for v in pos_by_class.values())} положительных, {len(neg_imgs)} отрицательных"
    )
    detector.set_references(pos_by_class=pos_by_class, neg_imgs=neg_imgs)

    # Детекция
    print("[INFO] Запуск детекции …")
    search_paths = sorted(args.search_dir.glob("*"))
    for img_path in search_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Пропускаю {img_path}: {e}")
            continue

        results = detector.detect(
            img,
            heatmap_threshold=args.heatmap_thresh,
            class_threshold=args.class_thresh,
        )

        print("\n=== Result for", img_path.name, "===")
        if not results:
            print("No detections")
        else:
            for det in results:
                print(
                    f"class_id={det.class_id}, score={det.score:.3f}, area={det.area}, bbox={det.bbox}"
                )


if __name__ == "__main__":
    main()
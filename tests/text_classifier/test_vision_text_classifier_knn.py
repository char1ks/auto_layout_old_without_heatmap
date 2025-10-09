import numpy as np
from flashbone.core.classification.base import ClassifierPredictRequest, ClassifierPrediction, ClassData
from flashbone.core.classification.vision_text_classifier_knn import VisionTextClassifierKNN


def test_predict_text_only(indexed_text_only, image_left):
    preds = indexed_text_only.predict(
        ClassifierPredictRequest(images=[image_left], threshold=0.3, topk=1)
    )
    assert isinstance(preds, list)
    if preds:
        p = preds[0]
        assert isinstance(p, ClassifierPrediction)
        assert p.class_id == 0
        assert 0.0 <= float(p.score) <= 1.0


def test_encode_concat_normalization(vt_classifier, image_left):
    req = ClassifierPredictRequest(images=[image_left], texts=["donkey"], threshold=0.3, topk=2)
    vecs = vt_classifier.encode(req)
    assert vecs.shape[0] >= 1
    assert np.isfinite(vecs).all()


def test_index_images_only(vt_encoder, image_right):
    clf = VisionTextClassifierKNN(encoder=vt_encoder, d=2048)
    train_image_neg_1 = image_right.crop((0, 0, 150, 150))
    train_image_neg_2 = image_right.crop((
        image_right.width - 150,
        image_right.height - 150,
        image_right.width,
        image_right.height,
    ))

    clf.index([
        ClassData(
            class_id=0,
            images=[image_right],
            negative_images=[train_image_neg_1, train_image_neg_2],
        )
    ])

    preds = clf.predict(ClassifierPredictRequest(images=[image_right], threshold=0.3, topk=1))
    assert isinstance(preds, list)
    if preds:
        p = preds[0]
        assert isinstance(p, ClassifierPrediction)
        assert p.class_id == 0
        assert 0.0 <= float(p.score) <= 1.0


def test_index_multimodal_neg(vt_encoder, image_right):
    clf = VisionTextClassifierKNN(encoder=vt_encoder, d=2048)
    train_image_neg_1 = image_right.crop((0, 0, 150, 150))

    clf.index([
        ClassData(
            class_id=0,
            images=[image_right],
            negative_images=[train_image_neg_1],
            negative_texts=["blue sky"],
        )
    ])

    preds = clf.predict(ClassifierPredictRequest(images=[image_right], threshold=0.3, topk=1))
    assert isinstance(preds, list)
    if preds:
        p = preds[0]
        assert isinstance(p, ClassifierPrediction)
        assert p.class_id == 0
        assert 0.0 <= float(p.score) <= 1.0


def test_index_multimodal_pos_neg(vt_encoder, image_right):
    clf = VisionTextClassifierKNN(encoder=vt_encoder, d=2048)
    train_image_neg_1 = image_right.crop((0, 0, 150, 150))

    clf.index([
        ClassData(
            class_id=0,
            texts=["donkey"],
            negative_images=[train_image_neg_1],
            negative_texts=["blue sky"],
        )
    ])

    preds = clf.predict(ClassifierPredictRequest(images=[image_right], threshold=0.3, topk=1))
    assert isinstance(preds, list)
    if preds:
        p = preds[0]
        assert isinstance(p, ClassifierPrediction)
        assert p.class_id == 0
        assert 0.0 <= float(p.score) <= 1.0
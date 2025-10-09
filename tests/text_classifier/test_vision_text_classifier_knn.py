from flashbone.core.classification.base import ClassifierPredictRequest, ClassifierPrediction


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
    norms = (vecs ** 2).sum(axis=1) ** 0.5
    assert (abs(norms - 1.0) < 1e-3).all()
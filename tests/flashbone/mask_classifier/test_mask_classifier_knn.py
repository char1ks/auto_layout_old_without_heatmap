import numpy as np
from flashbone.core.classification.base import ClassifierPredictRequest, ClassifierPrediction


def test_predict_with_mask(indexed_classifier, image_left, mask_left):
    req = ClassifierPredictRequest(
        images=[image_left],
        masks=[mask_left],
        threshold=0.4,
        mask_threshold=0.5,
        topk=1,
    )
    preds = indexed_classifier.predict(req)

    assert isinstance(preds, list)
    if preds:
        p = preds[0]
        assert isinstance(p, ClassifierPrediction)
        assert isinstance(p.class_id, int)
        assert 0.0 <= float(p.score) <= 1.0


def test_encode_normalization(classifier, image_left, mask_left):
    req = ClassifierPredictRequest(
        images=[image_left],
        masks=[mask_left],
        mask_threshold=0.5,
        topk=1,
    )
    vecs = classifier.encode(req)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 1
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
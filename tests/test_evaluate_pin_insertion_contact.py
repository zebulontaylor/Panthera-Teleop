import numpy as np
import pytest

from scripts.evaluate_pin_insertion_contact import check_heldout, error_vectors


def test_vector_error_requires_both_labels_and_preserves_direction():
    truth = np.array([[[.3, .4], [.2, .4]], [[.3, .4], [.2, .4]]])
    prediction = truth.copy()
    prediction[0] = prediction[0, ::-1]
    result = error_vectors(prediction, truth, np.array([[True, True], [True, False]]), np.array([[640, 480]] * 2))
    assert result['paired_labels'] == 1
    assert result['direction_cosine']['mean'] == pytest.approx(-1)
    assert result['pin_to_hole_vector_error_px']['mean'] == pytest.approx(128)


def test_unlabeled_contact_pair_is_not_reported_as_zero_error():
    x = np.zeros((1, 2, 2))
    result = error_vectors(x, x, np.array([[True, False]]), np.array([[640, 480]]))
    assert result['paired_labels'] == 0
    assert result['pin_to_hole_vector_error_px']['mean'] is None


def test_rejects_training_episode_and_selection_grasp():
    rows = [{'episode_id': 'held', 'session_id': 'g'}]
    with pytest.raises(ValueError, match='training/selection'):
        check_heldout(rows, {'adaptation': {'train_episode_ids': ['held']}})
    with pytest.raises(ValueError, match='grasp group'):
        check_heldout(rows, {'adaptation': {'manifest_metadata': {'episodes': [{'episode_id': 'different', 'session_id': 'g', 'split': 'validation'}]}}})

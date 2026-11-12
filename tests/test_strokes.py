import numpy as np
import pytest

from kanji_ml.strokes import drawing_to_model_input


def test_a_horizontal_stroke_becomes_a_soft_line_across_the_middle():
    out = drawing_to_model_input([[[30.0, 200.0], [330.0, 200.0]]])

    assert out.shape == (48, 48) and out.dtype == np.uint8
    rows = np.flatnonzero(out.any(axis=1))
    assert rows.min() >= 21 and rows.max() <= 26  # a line a few pixels thick, centred on row 23.5
    assert (out[:, 4:44] > 127).any(axis=0).all()  # it spans the 40-pixel box...
    assert not out[:, :3].any() and not out[:, 45:].any()  # ...and no further
    middle = out[:, 24]
    centre = int(np.argmax(middle))
    assert middle[centre] == 255
    assert (np.diff(middle[:centre + 1].astype(int)) >= 0).all()  # ink fades away from the centre line...
    assert (np.diff(middle[centre:].astype(int)) <= 0).all()
    assert ((middle > 0) & (middle < 200)).any()  # ...through soft grey, like ETL's scanned pen
    assert 2 <= (middle > 127).sum() <= 3  # about as thick as an ETL stroke


TREE = [  # 木, roughly as a finger draws it on the 370-point canvas
    [[50.0, 140.0], [320.0, 140.0]],
    [[185.0, 40.0], [185.0, 340.0]],
    [[180.0, 145.0], [60.0, 300.0]],
    [[190.0, 145.0], [320.0, 300.0]],
]


def test_where_and_how_big_the_drawing_is_does_not_matter():
    moved_and_scaled = [[[4 * x + 1000, 4 * y - 300] for x, y in stroke] for stroke in TREE]

    np.testing.assert_array_equal(drawing_to_model_input(moved_and_scaled), drawing_to_model_input(TREE))


@pytest.mark.parametrize("drawing", [[], [[]], [[], []]])
def test_an_empty_drawing_is_refused(drawing):
    with pytest.raises(ValueError, match="no points"):
        drawing_to_model_input(drawing)


def test_a_stroke_with_no_points_is_ignored():
    np.testing.assert_array_equal(drawing_to_model_input([[], *TREE, []]), drawing_to_model_input(TREE))


def test_a_tap_is_a_round_dot_in_the_middle():
    out = drawing_to_model_input([[[123.0, 45.0]]])

    assert out[23:25, 23:25].min() == 255  # brightest in the middle...
    assert not out[:4].any() and not out[44:].any() and not out[:, :4].any() and not out[:, 44:].any()
    np.testing.assert_array_equal(out, out.T)  # ...and the same in every direction
    np.testing.assert_array_equal(out, out[::-1, ::-1])


def test_going_over_part_of_a_stroke_again_does_not_change_the_drawing():
    stroke = [[0.0, 100.0], [200.0, 100.0]]
    retraced = [[120.0, 100.0], [200.0, 100.0]]  # the right part again: ink takes the stronger value, not the sum

    np.testing.assert_array_equal(drawing_to_model_input([stroke, retraced]), drawing_to_model_input([stroke]))

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landlab import RasterModelGrid


def test_add_full(at):
    grid = RasterModelGrid((4, 5))
    grid.add_full("temperature", 3, at=at)
    assert_array_equal(
        getattr(grid, f"at_{at}")["temperature"],
        np.full(grid.number_of_elements(at), 3),
    )


def test_add_field_with_units(at):
    grid = RasterModelGrid((4, 5))
    grid.add_empty("temperature", at=at, units="C")
    assert getattr(grid, f"at_{at}").units["temperature"] == "C"


def test_field_keys_is_list(at):
    grid = RasterModelGrid((4, 5))
    expected = []
    assert getattr(grid, f"at_{at}").keys() == expected
    for name in ["temperature", "elevation"]:
        grid.add_empty(name, at=at)
        expected.append(name)
        assert sorted(getattr(grid, f"at_{at}").keys()) == sorted(expected)


def test_add_field_at_node():
    """Add field at nodes."""
    grid = RasterModelGrid((4, 5))
    grid.add_field("z", np.arange(20), at="node")

    assert_array_equal(grid.at_node["z"], np.arange(20))


def test_add_field_without_at_keyword():
    """Test default is at nodes."""
    grid = RasterModelGrid((4, 5))
    grid.add_field("z", np.arange(20))

    assert_array_equal(grid.at_node["z"], np.arange(20))


def test_add_field_without_at():
    """Test raises error with wrong size array."""
    grid = RasterModelGrid((4, 5))
    with pytest.raises(ValueError):
        grid.add_field("z", np.arange(21))


def test_add_field_at_grid():
    """Test add field at grid."""
    grid = RasterModelGrid((4, 5))
    grid.at_grid["value"] = 1
    assert_array_equal(1, grid.at_grid["value"].size)


def test_adding_field_at_grid_two_ways():
    """Test add field at grid two ways."""
    grid = RasterModelGrid((4, 5))
    grid.at_grid["value_1"] = 1
    grid.add_field("value_2", 1, at="grid")
    assert_array_equal(grid.at_grid["value_1"], grid.at_grid["value_2"])


def test_add_ones_zeros_empty_to_at_grid():
    """Test different add methods for keyword at='grid'"""
    grid = RasterModelGrid((4, 5))
    with pytest.raises(ValueError):
        grid.add_zeros("value", at="grid")
    with pytest.raises(ValueError):
        grid.add_empty("value", at="grid")
    with pytest.raises(ValueError):
        grid.add_ones("value", at="grid")


def test_ones_zeros_empty_to_at_grid():
    """Test get array with field size methods for keyword at='grid'"""
    grid = RasterModelGrid((4, 5))
    with pytest.raises(ValueError):
        grid.zeros(at="grid")
    with pytest.raises(ValueError):
        grid.empty(at="grid")
    with pytest.raises(ValueError):
        grid.ones(at="grid")

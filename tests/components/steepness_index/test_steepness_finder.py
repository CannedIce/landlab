import pytest

from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from landlab.components import SteepnessFinder


def test_route_to_multiple_error_raised():
    mg = RasterModelGrid((10, 10))
    z = mg.add_zeros("topographic__elevation", at="node")
    z += mg.x_of_node + mg.y_of_node
    fa = FlowAccumulator(mg, flow_director="MFD")
    fa.run_one_step()

    with pytest.raises(NotImplementedError):
        SteepnessFinder(mg)

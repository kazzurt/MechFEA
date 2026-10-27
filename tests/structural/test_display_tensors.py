import numpy as np
from structural.results import result_display_field


def test_principal_and_explicit_reference_components():
    # Rotated principal axes: eigenvalues are 30, 10, -5 MPa.
    tensor=np.array([[[20.,10.,0.],[10.,20.,0.],[0.,0.,-5.]]])
    data={'stress_tensor_MPa_nodal_display':tensor.reshape(1,9),
          'physical_stress_valid':np.array([1]),'displacement_mm':np.array([[3.,-4.,0.]])}
    np.testing.assert_allclose(result_display_field(data,'principal_max')['values'],[30.])
    np.testing.assert_allclose(result_display_field(data,'principal_min')['values'],[-5.])
    np.testing.assert_allclose(result_display_field(data,'sxy')['values'],[10.])
    np.testing.assert_allclose(result_display_field(data,'uy')['values'],[-4.])
    data['physical_stress_valid'][:]=0
    assert result_display_field(data,'principal_max')['name'] is None

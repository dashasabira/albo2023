from albo.acquisition import OptimizationViaALBO


def test_my_acquisition():
    my_acq = MyAcquisition()
    my_acq.forward()

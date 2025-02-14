import numpy as np

from ch08_deep_learning.a_deep_2d_net import deep_2d_net_config
from ch08_deep_learning.test_a_deep_2d_net import train_and_test_deep_conv_net


def test_deep_conv_net_with_high_float_pricision() -> None:
    # the default float type is np.float32
    train_and_test_deep_conv_net(
        net_config=deep_2d_net_config(),
        target_float_dtype=np.float64,
        data_num_for_train=5000,
    )

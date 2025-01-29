from ch08_deep_learning.a_deep_2d_net import deep_2d_net_config
from ch08_deep_learning.test_a_deep_2d_net import train_and_test_deep_conv_net


def test_deep_conv_net_with_data_augmentation() -> None:
    # set save_params to True to save the parameters
    train_and_test_deep_conv_net(
        net_config=deep_2d_net_config(), augmente_data=True, save_params=False
    )

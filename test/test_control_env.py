def test_reset_control_env():
    from control_env.torchrl_control_env import JSBSimControlEnv

    env = JSBSimControlEnv()
    td_reset = env.reset()
    assert td_reset is not None

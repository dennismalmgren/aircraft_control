import torch

def test_create_sim():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cpu"))
    exec.allocate()

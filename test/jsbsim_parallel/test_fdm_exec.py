import torch
import os

def test_create_sim():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cpu"))
    exec.allocate()

def test_load_model():
    from jsbsim_parallel.fdm_exec import FDMExec
    #
    #self.jsbsim_exec = jsbsim.FGFDMExec(self.jsbsim_module_dir)
    #self.jsbsim_exec.load_model(self.aircraft_model)
        
    jsbsim_root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../py_modules/JSBSim')

    exec = FDMExec(torch.device("cpu"))
    exec.SetRootDir(jsbsim_root_path)
    exec.SetAircraftPath("aircraft")
    exec.SetSystemsPath("systems")
    exec.SetEnginePath("engine")
    exec.LoadModel("f16")

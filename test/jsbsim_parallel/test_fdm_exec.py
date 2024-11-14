import torch
import os

def test_create_sim():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cpu"))

def test_create_sim_gpu():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cuda:0"))

def test_create_sim_batch_1d():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cpu"), batch_size=torch.Size([2]))


def test_create_sim_batch_1d_gpu():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cuda:0"), batch_size=torch.Size([2]))

def test_create_sim_batch_2d():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cpu"), batch_size=torch.Size([2, 2]))

def test_create_sim_batch_2d_gpu():
    from jsbsim_parallel.fdm_exec import FDMExec
    exec = FDMExec(torch.device("cuda:0"), batch_size=torch.Size([2, 2]))


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

def test_load_model_batch_1d():
    from jsbsim_parallel.fdm_exec import FDMExec
    #
    #self.jsbsim_exec = jsbsim.FGFDMExec(self.jsbsim_module_dir)
    #self.jsbsim_exec.load_model(self.aircraft_model)
        
    jsbsim_root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../py_modules/JSBSim')

    exec = FDMExec(torch.device("cpu"),batch_size=torch.Size([2]))
    exec.SetRootDir(jsbsim_root_path)
    exec.SetAircraftPath("aircraft")
    exec.SetSystemsPath("systems")
    exec.SetEnginePath("engine")
    exec.LoadModel("f16")

def test_load_model_batch_1d_gpu():
    from jsbsim_parallel.fdm_exec import FDMExec
    #
    #self.jsbsim_exec = jsbsim.FGFDMExec(self.jsbsim_module_dir)
    #self.jsbsim_exec.load_model(self.aircraft_model)
        
    jsbsim_root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../py_modules/JSBSim')

    exec = FDMExec(torch.device("cuda:0"),batch_size=torch.Size([2]))
    exec.SetRootDir(jsbsim_root_path)
    exec.SetAircraftPath("aircraft")
    exec.SetSystemsPath("systems")
    exec.SetEnginePath("engine")
    exec.LoadModel("f16")
    #3324 mb before
    #3607 mb batch size 1 (283 mb)
    #3620 mb batch size 2 (296 mb)
    #3629 mb batch size 10 (305 mb)
    #3760 mb batch size 1000 (436 mb)

    print('ok')
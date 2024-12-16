from clib_interface.clib_exec import CLibExec

def main():
    exec = CLibExec()
    for i in range(3):
        state = exec.get_simulator_state()
        print(state.Z)
        exec.step()

if __name__=="__main__":
    main()

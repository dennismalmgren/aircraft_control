#ifndef _MY_SIM_INTEGRATION_LIB_H_
#define _MY_SIM_INTEGRATION_LIB_H_
        // ('X', ctypes.c_float),
        // ('Y', ctypes.c_float),
        // ('Z', ctypes.c_float),        
        // ('V_X', ctypes.c_float),
        // ('V_Y', ctypes.c_float),
        // ('V_Z', ctypes.c_float),        
        // ('SPEED', ctypes.c_float),
        // ('PHI', ctypes.c_float),
        // ('THETA', ctypes.c_float),
        // ('PSI', ctypes.c_float)



extern "C" {
    struct sim_state {
        float X;
        float Y;
        float Z;
        float V_X;
        float V_Y;
        float V_Z;
        float SPEED;
        float PHI;
        float THETA;
        float PSI;
    };

    void init_sim(double x, double y, float z, float speed, float heading);
    void set_inputs(float roll, float pitch, float throttle);
}

#endif
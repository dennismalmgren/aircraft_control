#ifndef _MY_SIM_INTEGRATION_LIB_H_
#define _MY_SIM_INTEGRATION_LIB_H_


extern "C" {
    struct sim_state {
        float X;
        float Y;
        float Z;

        float V_AEX;
        float V_AEY;
        float V_AEZ;

        float V_EX;
        float V_EY;
        float V_EZ;

        float A_EX;
        float A_EY;
        float A_EZ;

        float MY;
        float GAMMA;
        float CHI;
        float ALPHA;
        float BETA;
        float MACH;

        float PSI;
        float THETA;
        float PHI;

        float P;
        float Q;
        float R;

        float PDOT;
        float QDOT;
        float RDOT;

        float FUEL;
    };

    void init_sim(double x, double y, float z, float speed, float heading); // todo: add more
    void set_inputs(float roll, float pitch, float throttle);
    void get_state(sim_state* state);
    void step();
}

#endif
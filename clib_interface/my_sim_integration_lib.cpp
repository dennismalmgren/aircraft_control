#include "my_sim_integration_lib.h"
extern "C"  {

// TODO: Call lib
void init_sim(double x, double y, float z, float speed, float heading) {
 // set stuff according to mail

}

void set_inputs(float roll, float pitch, float throttle) {

}

void get_state(sim_state* state) {
    state->X = state->X + 1;
    state->Z = 10000.0;
}

void step() {

}

}

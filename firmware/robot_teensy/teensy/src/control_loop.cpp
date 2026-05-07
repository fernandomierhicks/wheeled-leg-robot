#include "control_loop.h"
#include "config.h"
#include "robot_state.h"

RobotState g_state = {};

static void isr_500hz() {
    // read sensors → compute → write actuators
}

void controlLoop_init() {
    // TODO: arm IntervalTimer at 1 000 000 / CONTROL_HZ µs
}

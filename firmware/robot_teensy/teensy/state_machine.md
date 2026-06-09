# State Machine

```mermaid
stateDiagram-v2
    [*] --> STARTUP

    STARTUP --> STANDBY : IMU NOMINAL + both motors heard
    STARTUP --> ESTOP   : IMU ERROR — OR — motors silent after 2 s

    STANDBY --> ESTOP   : hip feedback lost (CAN timeout 20 ms)
    STANDBY --> MANUAL  : GUI "Enter Manual"
    MANUAL  --> ESTOP   : hip feedback lost (CAN timeout 20 ms)
    MANUAL  --> STANDBY : GUI "Exit Manual"

    state STARTUP {
        direction LR
        s1 : imu_init()<br/>hip_motors_init()<br/>hip_motors_enter_mit()<br/>LED white breathe<br/>waits for IMU NOMINAL + both motors reply
    }

    state STANDBY {
        direction LR
        s2 : hip MIT active<br/>500 Hz — current pos + zero torque ping<br/>motors reply with position on every frame<br/>50 Hz telemetry TX<br/>no torque output
    }

    state MANUAL {
        direction LR
        s3 : processes one pending HipCmd per tick<br/>• ENABLE  → enter_mit()<br/>• DISABLE → exit_mit()<br/>• ZERO    → zero encoders<br/>• MIT     → send position/torque cmd<br/>control loop owns CAN frames<br/>50 Hz telemetry TX<br/><br/>STANDBY→MANUAL detail<br/>① GUI sends CMD_ID_SET_MODE / STATE_MANUAL<br/>② on_command() sets s_req_manual = true<br/>③ next stateMachine_update() (2 ms) fires req_manual()<br/>④ on_manual() runs — dispatches g_hip_cmd if pending<br/>⑤ keepalive CAN frames stop — GUI must send MIT cmds<br/>⑥ poll() still re-enters MIT every 3 s as safety net<br/>⑦ hip_fault() checked every tick → ESTOP if CAN silent
    }

    state ESTOP {
        direction LR
        s4 : LED red blink<br/>no exit transition — firmware reset required<br/>no hip commands sent
    }

    note right of STANDBY : CALIBRATION and RUNNING defined<br/>in RobotStateEnum but not yet<br/>wired into the state machine
```

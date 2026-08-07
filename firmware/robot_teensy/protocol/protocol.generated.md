# Generated robot protocol reference

Do not edit; generated from `protocol/schema.json`.

## States

| ID | Symbol | Description |
| ---: | --- | --- |
| `0x00` | `STATE_STARTUP` |  |
| `0x01` | `STATE_CALIBRATION` |  |
| `0x02` | `STATE_STANDBY` |  |
| `0x03` | `STATE_RUNNING` |  |
| `0x04` | `STATE_ESTOP` |  |
| `0x05` | `STATE_MANUAL` |  |
| `0x06` | `STATE_CMD_REJECT` |  |
| `0x07` | `STATE_JUMPING` |  |
| `0x08` | `STATE_STANDING_UP` |  |
| `0x09` | `STATE_DISARMING` |  |

## Faults

| ID | Symbol | Description |
| ---: | --- | --- |
| `0x00` | `FAULT_NONE` |  |
| `0x01` | `FAULT_IMU_ERROR` | IMU reported ERROR during startup |
| `0x02` | `FAULT_HIP_INIT_TIMEOUT` | no CAN reply from hip motors within 2 s of boot |
| `0x03` | `FAULT_HIP_FEEDBACK_LOST` | hip CAN feedback timed out during operation |
| `0x04` | `FAULT_HIP_LARGE_POS_CMD` | commanded position jump exceeded MAX_HIP_DELTA_RAD |
| `0x05` | `FAULT_CALIBRATION_TIMEOUT` | retract-switch homing safety check failed |
| `0x06` | `FAULT_HUMAN_ESTOP` | ESTOP requested by user via GUI button or radio |
| `0x08` | `FAULT_PITCH_WATCHDOG` | |pitch| > 50° for > 200 ms |
| `0x09` | `FAULT_WHEEL_RUNAWAY` | wheel velocity exceeded 2× soft governor limit |
| `0x0A` | `FAULT_IMU_LOST` | IMU left NOMINAL while RUNNING/JUMPING (silence or heavy loss) |
| `0x0B` | `FAULT_WHEEL_FEEDBACK_LOST` | wheel encoder timeout or ODrive error during operation |
| `0x0C` | `FAULT_WHEEL_INIT_TIMEOUT` | no CAN reply from wheel motors within 2 s of boot |
| `0x0D` | `FAULT_STANDUP_FAILED` | standup denied (pitch out of recoverable range) or exhausted retries/diverged |
| `0x0E` | `FAULT_ROLL_WATCHDOG` | |roll| > roll_watchdog_limit for > 200 ms (lateral tip guard) |
| `0x0F` | `FAULT_JUMP_TIMEOUT` | jump sequence overran its computed phase budget without reaching JP_DONE |

## Commands

| ID | Symbol | Description |
| ---: | --- | --- |
| `0x01` | `CMD_ID_SET_MODE` | payload: uint8_t target_state (RobotStateEnum) |
| `0x02` | `CMD_ID_PING` | payload: none — GUI heartbeat; feeds the MANUAL GUI watchdog only |
| `0x05` | `CMD_ID_HIP` | payload: uint8_t motor_id, uint8_t sub_cmd [, 5×float] |
| `0x06` | `CMD_ID_REBOOT` | payload: none — triggers a full MCU reset (reruns setup()) |
| `0x07` | `CMD_ID_WHEEL` | payload: uint8_t sub_cmd [, data] |
| `0x08` | `CMD_ID_SET_TELEM_TRANSPORT` | PC→ESP32 only (intercepted, never forwarded to Teensy): |
| `0x10` | `CMD_ID_PARAM_SET` | payload: uint16_t param_id, float value  (6 bytes after cmd_id) |
| `0x11` | `CMD_ID_PARAM_GET` | payload: uint16_t param_id  (0xFFFF = dump all) |
| `0x12` | `CMD_ID_LOG` | payload: uint8_t sub_cmd [, args] — high-datarate SD logging |
| `0x13` | `CMD_ID_PARAM_RESET_DEFAULTS` | payload: none — reverts all writable params to |
| `0x14` | `CMD_ID_TEST_INJECT_CORRUPT` | TEST ONLY (Phase 9, UARTplat.md stress testing): |

## Parameters

| ID | Symbol | Name | Default | Range | Flags |
| ---: | --- | --- | ---: | --- | --- |
| `0x0000` | `PARAM_IMU_ENABLE` | `imu_enable` | 1 | 0 … 1 | persistent |
| `0x0003` | `PARAM_BUZZER_VOLUME` | `buzzer_volume` | 1 | 0 … 1 | persistent |
| `0x0004` | `PARAM_LED_ENABLE` | `led_enable` | 1 | 0 … 1 | persistent |
| `0x0005` | `PARAM_HIP_L_ENABLE` | `hip_l_enable` | 0 | 0 … 1 | persistent |
| `0x0006` | `PARAM_HIP_R_ENABLE` | `hip_r_enable` | 0 | 0 … 1 | persistent |
| `0x0007` | `PARAM_WHEEL_L_ENABLE` | `wheel_l_enable` | 1 | 0 … 1 | persistent |
| `0x0008` | `PARAM_WHEEL_R_ENABLE` | `wheel_r_enable` | 1 | 0 … 1 | persistent |
| `0x0009` | `PARAM_WATCHDOG_ENABLE` | `watchdog_enable` | 0 | 0 … 1 | persistent |
| `0x000A` | `PARAM_LOOP_PROFILE_ENABLE` | `loop_profile_enable` | 0 | 0 … 1 | - |
| `0x0201` | `PARAM_HIP_RUNNING_KP` | `hip_running_kp` | 5 | 0 … 100 | persistent |
| `0x0202` | `PARAM_HIP_RUNNING_KD` | `hip_running_kd` | 0.5 | 0 … 5 | persistent |
| `0x0203` | `PARAM_HIP_RUNNING_TFF_RET` | `hip_running_tff_ret` | 0 | -8 … 8 | persistent |
| `0x0204` | `PARAM_HIP_RUNNING_RAMP_TIME_S` | `hip_running_ramp_s` | 2 | 0 … 10 | persistent |
| `0x0205` | `PARAM_HIP_CMD_RATE_LIMIT_PER_S` | `hip_cmd_rate_lim` | 0.2 | 0 … 2 | persistent |
| `0x0206` | `PARAM_HIP_RUNNING_TFF_EXT` | `hip_running_tff_ext` | 0 | -8 … 8 | persistent |
| `0x0200` | `PARAM_ESTOP_HIP_DISABLE` | `estop_hip_disable` | 1 | 0 … 1 | persistent |
| `0x0120` | `PARAM_CALIB_SEEK_SPEED` | `calib_seek_speed` | 0.17453 | 0.01 … 0.5 | persistent |
| `0x0121` | `PARAM_CALIB_MOVE_SPEED` | `calib_move_speed` | 0.174533 | 0.01 … 0.5 | persistent |
| `0x0122` | `PARAM_CALIB_SEEK_KP` | `calib_seek_kp` | 16 | 0.1 … 100 | persistent |
| `0x0123` | `PARAM_CALIB_KD` | `calib_kd` | 0.05 | 0 … 5 | persistent |
| `0x0124` | `PARAM_CALIB_SEEK_TORQUE_LIMIT_NM` | `calib_seek_trq_lim` | 0.3 | 0.05 … 2 | persistent |
| `0x0125` | `PARAM_CALIB_MOVE_TORQUE_LIMIT_NM` | `calib_move_trq_lim` | 0.6 | 0.05 … 4 | persistent |
| `0x0126` | `PARAM_CALIB_BACKOFF_RAD` | `calib_backoff_rad` | 0.0872665 | 0.01 … 0.35 | persistent |
| `0x0127` | `PARAM_CALIB_RANGE_L_RAD` | `calib_range_l_rad` | 1.5708 | 0.1 … 3.14159 | persistent |
| `0x0128` | `PARAM_CALIB_RANGE_R_RAD` | `calib_range_r_rad` | 1.5708 | 0.1 … 3.14159 | persistent |
| `0x0129` | `PARAM_CALIB_SEEK_TIMEOUT_S` | `calib_seek_timeout` | 30 | 1 … 60 | persistent |
| `0x012A` | `PARAM_CALIB_RELEASE_TIMEOUT_S` | `calib_release_to` | 5 | 0.5 … 30 | persistent |
| `0x012B` | `PARAM_CALIB_MAX_SEEK_TRAVEL_RAD` | `calib_max_seek_rad` | 2 | 0.1 … 6.28319 | persistent |
| `0x012C` | `PARAM_CALIB_MAX_RELEASE_TRAVEL_RAD` | `calib_max_rel_rad` | 0.5 | 0.05 … 1.5708 | persistent |
| `0x012D` | `PARAM_CALIB_TORQUE_TRIP_MS` | `calib_trq_trip_ms` | 50 | 0 … 500 | persistent |
| `0x012E` | `PARAM_CALIB_MOVE_KP` | `calib_move_kp` | 32 | 0.1 … 100 | persistent |
| `0x012F` | `PARAM_CALIB_RAMPDOWN_TIME_S` | `calib_rampdown_s` | 2 | 0 … 10 | persistent |
| `0x0131` | `PARAM_CALIB_BYPASS_EN` | `calib_bypass_en` | 0 | 0 … 1 | persistent |
| `0x0300` | `PARAM_WM_ENC_TIMEOUT_MS` | `wm_enc_timeout_ms` | 20 | 5 … 500 | persistent |
| `0x0301` | `PARAM_WM_VEL_SLEW_MAX` | `wm_vel_slew_max` | 1500 | 0 … 20000 | persistent |
| `0x0400` | `PARAM_LQR_ENABLE` | `lqr_enable` | 1 | 0 … 1 | persistent |
| `0x0401` | `PARAM_SIM_PITCH_RAD` | `sim_pitch_rad` | 0 | -1.5708 … 1.5708 | - |
| `0x0402` | `PARAM_LQR_TORQUE_LIMIT` | `lqr_torque_limit` | 0.1 | 0 … 7 | readonly, command |
| `0x0403` | `PARAM_WHEEL_VEL_LIMIT_TURNS_S` | `wm_vel_limit` | 3 | 1 … 20 | persistent |
| `0x0404` | `PARAM_VEL_PI_EN` | `vel_pi_en` | 1 | 0 … 1 | persistent |
| `0x0405` | `PARAM_VEL_PI_KP` | `vel_pi_kp` | 0.2 | 0 … 5 | persistent |
| `0x0406` | `PARAM_VEL_PI_KI` | `vel_pi_ki` | 0.1 | 0 … 5 | persistent |
| `0x0407` | `PARAM_VEL_PI_KFF` | `vel_pi_kff` | 0.1049 | 0 … 1 | persistent |
| `0x0409` | `PARAM_VEL_PI_RATE_LIM` | `vel_pi_rate_lim` | 1.745 | 0.1 … 10 | persistent |
| `0x040A` | `PARAM_VEL_PI_INT_MAX` | `vel_pi_int_max` | 1 | 0.1 … 5 | persistent |
| `0x040B` | `PARAM_V_CMD_MS` | `v_cmd_ms` | 0 | -2 … 2 | command |
| `0x040C` | `PARAM_YAW_PI_EN` | `yaw_pi_en` | 1 | 0 … 1 | persistent |
| `0x040D` | `PARAM_YAW_PI_KP` | `yaw_pi_kp` | 0.2 | 0 … 5 | persistent |
| `0x040E` | `PARAM_YAW_PI_KI` | `yaw_pi_ki` | 0.1 | 0 … 5 | persistent |
| `0x040F` | `PARAM_YAW_PI_TORQUE_MAX` | `yaw_pi_torque_max` | 0.2 | 0 … 3 | persistent |
| `0x0410` | `PARAM_YAW_PI_INT_MAX` | `yaw_pi_int_max` | 0.5 | 0 … 3 | persistent |
| `0x0411` | `PARAM_OMEGA_CMD_RDS` | `omega_cmd_rds` | 0 | -4 … 4 | command |
| `0x0412` | `PARAM_FF1_ALPHA` | `ff1_alpha` | 0 | 0 … 1 | - |
| `0x0413` | `PARAM_FF2_ALPHA` | `ff2_alpha` | 0 | 0 … 1 | - |
| `0x0414` | `PARAM_FF1_KT_HIP` | `ff1_kt_hip` | 1 | 0 … 5 | readonly |
| `0x0415` | `PARAM_JUMP_ENABLE` | `jump_enable` | 0 | 0 … 1 | persistent |
| `0x0416` | `PARAM_JUMP_TORQUE_MAX` | `jump_torque_max` | 0 | 0 … 8 | persistent |
| `0x0417` | `PARAM_JUMP_CROUCH_TIME_S` | `jump_crouch_time` | 0.3 | 0.05 … 1 | persistent |
| `0x0418` | `PARAM_JUMP_RAMP_UP_S` | `jump_ramp_up` | 0.05 | 0.005 … 0.5 | persistent |
| `0x0419` | `PARAM_JUMP_RAMP_DOWN_RAD` | `jump_ramp_down` | 0.08 | 0.01 … 0.5 | persistent |
| `0x041A` | `PARAM_JUMP_OMEGA_MAX` | `jump_omega_max` | 12.3077 | 2 … 20 | persistent |
| `0x041B` | `PARAM_JUMP_HARDSTOP_MARGIN` | `jump_hs_margin` | 0.06 | 0.01 … 0.5 | persistent |
| `0x041C` | `PARAM_JUMP_KP` | `jump_kp` | 80 | 0 … 500 | persistent |
| `0x041D` | `PARAM_JUMP_KD` | `jump_kd` | 1 | 0 … 5 | persistent |
| `0x041E` | `PARAM_JUMP_EXTEND_KD` | `jump_ext_kd` | 0.1 | 0 … 5 | persistent |
| `0x041F` | `PARAM_JUMP_EXTEND_TIMEOUT_S` | `jump_ext_timeout` | 0.15 | 0.05 … 1 | persistent |
| `0x0420` | `PARAM_ENABLE_SIM_PITCH_RAD` | `enable_sim_pitch` | 0 | 0 … 1 | - |
| `0x0421` | `PARAM_SIM_PITCH_RATE_RAD_S` | `sim_pitch_rate` | 0 | -10 … 10 | - |
| `0x0422` | `PARAM_ENABLE_SIM_PITCH_RATE` | `enable_sim_prate` | 0 | 0 … 1 | - |
| `0x0423` | `PARAM_PITCH_WATCHDOG_ENABLE` | `pitch_watchdog_en` | 1 | 0 … 1 | - |
| `0x0424` | `PARAM_LQR_K_PITCH_RET` | `lqr_k_pitch_ret` | -0.3 | -20 … 0 | persistent |
| `0x0425` | `PARAM_LQR_K_RATE_RET` | `lqr_k_rate_ret` | -0.1 | -5 … 0 | persistent |
| `0x0426` | `PARAM_LQR_K_PITCH_EXT` | `lqr_k_pitch_ext` | -0.3 | -15 … 0 | persistent |
| `0x0427` | `PARAM_LQR_K_RATE_EXT` | `lqr_k_rate_ext` | -0.1 | -5 … 0 | persistent |
| `0x0428` | `PARAM_LQR_K_VEL` | `lqr_k_vel` | -0.007 | -1 … 0 | persistent |
| `0x0429` | `PARAM_RUNNING_WHEEL_BYPASS_EN` | `run_wheel_bypass_en` | 0 | 0 … 1 | - |
| `0x042A` | `PARAM_ALPHA_FORCE_RETRACTED_EN` | `alpha_force_ret_en` | 0 | 0 … 1 | - |
| `0x042B` | `PARAM_GUI_MOTION_CTRL_EN` | `gui_motion_ctrl_en` | 0 | 0 … 1 | - |
| `0x042C` | `PARAM_STANDUP_ENABLE` | `standup_enable` | 0 | 0 … 1 | persistent |
| `0x042D` | `PARAM_STANDUP_MAX_PITCH_FWD_RAD` | `standup_pitch_fwd` | 0.6 | 0 … 1.4 | persistent |
| `0x042E` | `PARAM_STANDUP_MAX_PITCH_BWD_RAD` | `standup_pitch_bwd` | 0.6 | 0 … 1.4 | persistent |
| `0x0431` | `PARAM_STANDUP_CROUCH_TIME_S` | `standup_crouch_time` | 0.3 | 0.05 … 2 | persistent |
| `0x0432` | `PARAM_STANDUP_K_PITCH` | `standup_k_pitch` | 0 | 0 … 60 | persistent |
| `0x0433` | `PARAM_STANDUP_K_RATE` | `standup_k_rate` | 0 | 0 … 15 | persistent |
| `0x0434` | `PARAM_STANDUP_TORQUE_LIMIT` | `standup_torque_lim` | 0 | 0 … 7 | persistent |
| `0x0435` | `PARAM_STANDUP_WHEEL_VEL_LIMIT_TURNS_S` | `standup_vel_limit` | 3 | 1 … 20 | persistent |
| `0x0436` | `PARAM_STANDUP_CAPTURE_PITCH_RAD` | `standup_cap_pitch` | 0.12 | 0.02 … 0.4 | persistent |
| `0x0437` | `PARAM_STANDUP_CAPTURE_RATE_RADS` | `standup_cap_rate` | 1 | 0.1 … 5 | persistent |
| `0x0438` | `PARAM_STANDUP_CAPTURE_HOLD_S` | `standup_cap_hold` | 0.15 | 0.02 … 1 | persistent |
| `0x0439` | `PARAM_STANDUP_ATTEMPT_TIMEOUT_S` | `standup_timeout` | 1.5 | 0.2 … 5 | persistent |
| `0x043A` | `PARAM_STANDUP_MAX_RETRIES` | `standup_max_retries` | 2 | 0 … 10 | persistent |
| `0x043B` | `PARAM_STANDUP_RETRY_PAUSE_S` | `standup_retry_pause` | 0.3 | 0 … 3 | persistent |
| `0x043C` | `PARAM_LQR_PITCH_TRIM_RET` | `lqr_pitch_trim_ret` | -0.14 | -0.3491 … 0.3491 | persistent |
| `0x043D` | `PARAM_LQR_PITCH_TRIM_EXT` | `lqr_pitch_trim_ext` | 0 | -0.3491 … 0.3491 | persistent |
| `0x043E` | `PARAM_VEL_PI_THETA_MAX_FWD_RET` | `theta_max_fwd_ret` | 0.5235988 | 0.1 … 0.6981317 | persistent |
| `0x043F` | `PARAM_VEL_PI_THETA_MAX_BWD_RET` | `theta_max_bwd_ret` | 0.1745329 | 0.1 … 0.6981317 | persistent |
| `0x0440` | `PARAM_VEL_PI_THETA_MAX_FWD_EXT` | `theta_max_fwd_ext` | 0.5235988 | 0.1 … 0.6981317 | persistent |
| `0x0441` | `PARAM_VEL_PI_THETA_MAX_BWD_EXT` | `theta_max_bwd_ext` | 0.1745329 | 0.1 … 0.6981317 | persistent |
| `0x0442` | `PARAM_PITCH_WATCHDOG_FWD_RET` | `pitch_wd_fwd_ret` | 0.6981317 | 0 … 1.4 | persistent |
| `0x0443` | `PARAM_PITCH_WATCHDOG_BWD_RET` | `pitch_wd_bwd_ret` | 0.2617994 | 0 … 1.4 | persistent |
| `0x0444` | `PARAM_PITCH_WATCHDOG_FWD_EXT` | `pitch_wd_fwd_ext` | 0.6981317 | 0 … 1.4 | persistent |
| `0x0445` | `PARAM_PITCH_WATCHDOG_BWD_EXT` | `pitch_wd_bwd_ext` | 0.2617994 | 0 … 1.4 | persistent |
| `0x0446` | `PARAM_STANDUP_DIVERGE_PITCH_FWD_RAD` | `standup_div_fwd` | 1 | 0 … 1.4 | persistent |
| `0x0447` | `PARAM_STANDUP_DIVERGE_PITCH_BWD_RAD` | `standup_div_bwd` | 1 | 0 … 1.4 | persistent |
| `0x0448` | `PARAM_LQR_BARRIER_K` | `lqr_barrier_k` | 0 | 0 … 60 | persistent |
| `0x0449` | `PARAM_LQR_BARRIER_THRESH_RET` | `lqr_barrier_th_ret` | 0.20944 | 0.1 … 0.6981317 | persistent |
| `0x044A` | `PARAM_LQR_BARRIER_THRESH_EXT` | `lqr_barrier_th_ext` | 0.20944 | 0.1 … 0.6981317 | persistent |
| `0x0500` | `PARAM_RADIO_HIP_CMD` | `radio_hip_cmd` | 0 | 0 … 1 | readonly, command |
| `0x0501` | `PARAM_RADIO_VEL_MAX` | `radio_vel_max` | 0.5 | 0 … 2 | readonly, command |
| `0x0502` | `PARAM_RADIO_YAW_MAX` | `radio_yaw_max` | 1 | 0 … 4 | readonly, command |
| `0x0503` | `PARAM_LIVE_TUNE_CH7_VAL` | `live_tune_ch7_val` | 0 | -2 … 4 | readonly, command |
| `0x051B` | `PARAM_LIVE_TUNE_CH8_VAL` | `live_tune_ch8_val` | 0 | -1 … 0.5 | readonly, command |
| `0x0510` | `PARAM_PROFILE_1_VEL_MAX` | `profile1_vel_max` | 0.2 | 0 … 2 | persistent |
| `0x0511` | `PARAM_PROFILE_1_YAW_MAX` | `profile1_yaw_max` | 0.5 | 0 … 4 | persistent |
| `0x0512` | `PARAM_PROFILE_1_TORQUE_LIM` | `profile1_torque_lim` | 0.1 | 0 … 7 | persistent |
| `0x0513` | `PARAM_PROFILE_2_VEL_MAX` | `profile2_vel_max` | 0.5 | 0 … 2 | persistent |
| `0x0514` | `PARAM_PROFILE_2_YAW_MAX` | `profile2_yaw_max` | 1 | 0 … 4 | persistent |
| `0x0515` | `PARAM_PROFILE_2_TORQUE_LIM` | `profile2_torque_lim` | 0.2 | 0 … 7 | persistent |
| `0x0516` | `PARAM_PROFILE_3_VEL_MAX` | `profile3_vel_max` | 1 | 0 … 2 | persistent |
| `0x0517` | `PARAM_PROFILE_3_YAW_MAX` | `profile3_yaw_max` | 2 | 0 … 4 | persistent |
| `0x0518` | `PARAM_PROFILE_3_TORQUE_LIM` | `profile3_torque_lim` | 0.3 | 0 … 7 | persistent |
| `0x0519` | `PARAM_ACTIVE_PROFILE` | `active_profile` | 0 | 0 … 2 | readonly, command |
| `0x051A` | `PARAM_LIVE_TUNE_LATCH` | `live_tune_latch` | 0 | 0 … 1 | command |
| `0x051C` | `PARAM_ROLL_CTRL_EN` | `roll_ctrl_en` | 0 | 0 … 1 | persistent |
| `0x051D` | `PARAM_ROLL_KP` | `roll_kp` | 1 | 0 … 20 | persistent |
| `0x051E` | `PARAM_ROLL_KD` | `roll_kd` | 0.1 | 0 … 5 | persistent |
| `0x052A` | `PARAM_ROLL_KI` | `roll_ki` | 0 | 0 … 5 | persistent |
| `0x052B` | `PARAM_ROLL_INT_MAX` | `roll_int_max` | 0.1 | 0 … 2 | persistent |
| `0x051F` | `PARAM_ROLL_OFFSET_MAX` | `roll_offset_max` | 0.15 | 0 … 0.6 | persistent |
| `0x0520` | `PARAM_ROLL_RATE_LIM` | `roll_rate_lim` | 0.5 | 0 … 5 | persistent |
| `0x0521` | `PARAM_ROLL_WATCHDOG_EN` | `roll_watchdog_en` | 0 | 0 … 1 | persistent |
| `0x0522` | `PARAM_ROLL_WATCHDOG_LIMIT` | `roll_watchdog_limit` | 0.35 | 0 … 1 | persistent |
| `0x0523` | `PARAM_HIP_ROLL_KP` | `hip_roll_kp` | 3 | 0 … 100 | persistent |
| `0x0524` | `PARAM_HIP_ROLL_KD` | `hip_roll_kd` | 0.5 | 0 … 5 | persistent |
| `0x0525` | `PARAM_ROLL_CMD_RAD` | `roll_cmd_rad` | 0 | -1 … 1 | readonly, command |
| `0x0526` | `PARAM_RADIO_ROLL_MAX` | `radio_roll_max` | 0.1 | 0 … 0.5 | readonly, command |
| `0x0527` | `PARAM_PROFILE_1_ROLL_MAX` | `profile1_roll_max` | 0.05 | 0 … 0.5 | persistent |
| `0x0528` | `PARAM_PROFILE_2_ROLL_MAX` | `profile2_roll_max` | 0.1 | 0 … 0.5 | persistent |
| `0x0529` | `PARAM_PROFILE_3_ROLL_MAX` | `profile3_roll_max` | 0.17 | 0 … 0.5 | persistent |

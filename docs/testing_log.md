7/19/2026

Succesfull balancing and steering. 


7/17/2026

## Calibration

Both legs with wheels. 

calib_l_seek_dir=-1
calib_r_seek_dir=+1

Retraction moves needs 
calib_stall_curr_btm to be 1A, to counteract the weight of the wheel motor + wheel. 
calib_stall_curr_top to be 5A, to allow holding the weight of the wheel motor while smoothly going "down" into the extended position. 


Hips need hip_running_kp=20 to be able to move up and down in running mode. 



---

7/12/2026

## Calibration

Right leg only, no wheel. Robot body clamped to desk. Opposite dynamics than real robot. Retractio is hard, extension easy (real robot would be backwards)

Retraction moves needs 
calib_stall_curr_btm to be 5A, to counteract the weight of the wheel motor. 
calib_stall_curr_top to be 4A, to allow holding the weight of the wheel motor while smoothly going "down" into the extended position. 

After calibration.Able to move hip up and down along with turning wheel at the same time. Both with sine waves. 

GOOD TO PROCEED TO WIRE OTHER LEG!!!






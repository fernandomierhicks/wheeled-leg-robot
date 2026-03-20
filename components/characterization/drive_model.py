"""
components/characterization/drive_model.py

Models the electrical and mechanical performance limits of the drive train
(Battery -> Controller -> Motor).

Useful for determining:
- Maximum torque vs speed (Torque Curve)
- Top speed under load
- Thermal heating power vs Mechanical power

Usage:
    python components/characterization/drive_model.py
"""

import numpy as np
import matplotlib.pyplot as plt

class Battery:
    def __init__(self, voltage_S, cells_S, capacity_Ah, c_rating, name="Lipo"):
        self.name = name
        self.voltage_nom = voltage_S * cells_S
        # Internal resistance approx: 1 / (Capacity * C_rating) or explicit
        # e.g., 5Ah * 50C = 250A max, approx 0.004 Ohm internal? 
        # Let's use a conservative estimate or calculating it:
        self.r_internal = 1.0 / (capacity_Ah * c_rating) if c_rating > 0 else 0.1

    def get_voltage(self, current_total):
        """Returns terminal voltage under load."""
        return max(0.0, self.voltage_nom - (self.r_internal * current_total))

class Motor:
    def __init__(self, name, kv, r_phase_ohm, l_phase_uH, i_max_peak):
        self.name = name
        self.kv = kv
        # Torque constant Kt [Nm/A] = 60 / (2*pi*KV)
        self.kt = 60 / (2 * np.pi * kv) if kv > 0 else 0
        self.r_phase = r_phase_ohm
        self.l_phase = l_phase_uH * 1e-6 # convert to Henry
        self.i_max = i_max_peak

class Controller:
    def __init__(self, name, i_max_peak, v_max_rating=56.0):
        self.name = name
        self.i_max = i_max_peak
        self.v_max_rating = v_max_rating

class DriveTrain:
    def __init__(self, motor: Motor, controller: Controller, battery: Battery):
        self.motor = motor
        self.controller = controller
        self.battery = battery
        self.system_i_max = min(motor.i_max, controller.i_max)

    def calculate_max_torque(self, speed_rpm):
        """
        Calculates max possible torque at a given speed, limited by:
        1. Current limit (Controller/Motor)
        2. Voltage limit (Battery sag - BackEMF - Impedance drop)
        """
        omega = speed_rpm * (2 * np.pi / 60.0)
        
        # Iterative solution because Battery Voltage depends on Current
        # Start assuming max current
        i_q = self.system_i_max
        
        for _ in range(5): # Convergence loop
            v_batt = self.battery.get_voltage(i_q)
            # Bus voltage limit (battery or controller rating)
            v_bus = min(v_batt, self.controller.v_max_rating)
            
            # Space Vector Modulation limit (max phase voltage amplitude)
            v_max_vector = v_bus / np.sqrt(3)
            
            # FOC Voltage equation magnitude (steady state, Id=0 for max torque/amp):
            # V^2 = (R*Iq + omega*Kt)^2 + (omega*L*Iq)^2
            # We solve for max Iq given V_max_vector
            
            R = self.motor.r_phase
            L = self.motor.l_phase
            Kt = self.motor.kt
            
            # (R^2 + w^2 L^2) I^2 + (2 R w Kt) I + (w^2 Kt^2 - Vmax^2) = 0
            a = R**2 + (omega * L)**2
            b = 2 * R * omega * Kt
            c = (omega * Kt)**2 - v_max_vector**2
            
            if abs(a) < 1e-9:
                i_voltage_limit = self.system_i_max
            else:
                delta = b**2 - 4*a*c
                if delta < 0:
                    i_voltage_limit = 0.0 # Cannot spin at this speed (BackEMF > Vbus)
                else:
                    i_voltage_limit = (-b + np.sqrt(delta)) / (2*a)
            
            # Clamp to system limits
            i_q_new = np.clip(i_voltage_limit, 0, self.system_i_max)
            
            if abs(i_q_new - i_q) < 0.1:
                i_q = i_q_new
                break
            i_q = i_q_new

        torque = i_q * self.motor.kt
        power_elec = i_q * v_batt # Approx DC power
        power_mech = torque * omega
        heat_watts = 1.5 * i_q**2 * self.motor.r_phase # 3-phase resistive heat
        
        return {
            "rpm": speed_rpm,
            "torque": torque,
            "current": i_q,
            "power_mech": power_mech,
            "heat_watts": heat_watts,
            "voltage_saturation": i_q < (self.system_i_max * 0.95) and i_q > 0.1
        }

def plot_performance(drive: DriveTrain):
    rpms = np.linspace(0, 6000, 100)
    results = [drive.calculate_max_torque(r) for r in rpms]
    
    torques = [r['torque'] for r in results]
    heats = [r['heat_watts'] for r in results]
    powers = [r['power_mech'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_title(f"Drive Performance: {drive.motor.name} w/ {drive.controller.name}\nBattery: {drive.battery.voltage_nom:.1f}V Nominal")
    ax1.set_xlabel("Speed (RPM)")
    ax1.set_ylabel("Torque (Nm)", color='tab:blue')
    ax1.plot(rpms, torques, color='tab:blue', linewidth=2, label="Max Torque")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Power (W)", color='tab:red')
    ax2.plot(rpms, heats, color='tab:red', linestyle='--', label="Heat Dissipation (W)")
    ax2.plot(rpms, powers, color='tab:green', linestyle='-', label="Mech Power (W)")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Mark saturation point
    sat_points = [r for r in results if r['voltage_saturation']]
    if sat_points:
        first_sat = sat_points[0]
        ax1.axvline(x=first_sat['rpm'], color='k', linestyle=':', alpha=0.5)
        ax1.text(first_sat['rpm'], max(torques)*0.9, f" Voltage Saturation\n @ {first_sat['rpm']:.0f} RPM", rotation=90, verticalalignment='top')

    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    
    # Motor: Maytech 5065 70KV (Baseline 1 Design)
    motor = Motor(
        name="Maytech 5065 70KV",
        kv=70,
        r_phase_ohm=0.080,  # Estimated (derived from R_eff=0.24 in sim_config)
        l_phase_uH=40.0,    # Estimated (higher inductance for lower KV)
        i_max_peak=50.0     # Rated max
    )
    
    # Controller: ODrive v3.6
    controller = Controller(
        name="ODrive v3.6",
        i_max_peak=60.0,    # Hardware limit (often thermal limited)
        v_max_rating=56.0
    )
    
    # Battery: 6S LiPo (22.2V Nom)
    battery = Battery(
        voltage_S=3.7,
        cells_S=6,
        capacity_Ah=5.0,
        c_rating=50
    )

    drive = DriveTrain(motor, controller, battery)
    
    print(f"Motor Kt: {drive.motor.kt:.4f} Nm/A")
    print(f"System Current Limit: {drive.system_i_max:.1f} A")
    print(f"Low Speed Torque: {drive.system_i_max * drive.motor.kt:.2f} Nm")
    
    plot_performance(drive)
import numpy as np
import matplotlib.pyplot as plt

# Define aerodynamic parameters from LiftCfg
air_density = 1.225         # kg/m^3
reference_area = 0.5        # m^2
CL0 = 0.2                 # Lift coefficient at alpha = 0
CL_alpha = 5.7            # Lift slope per radian

# Assume a constant speed (Va) for this example (e.g., 10 m/s)
Va = 10.0  # m/s

# Compute the dynamic pressure: q_dyn = 0.5 * rho * Va^2
q_dyn = 0.5 * air_density * Va**2

# Define a range for angle of attack (alpha) in radians
alpha = np.linspace(-1.5, 1.5, 300)  # Limits based on alpha_max in LiftCfg

# Compute the lift coefficient: CL = CL0 + CL_alpha * alpha
CL = CL0 + CL_alpha * alpha

# Compute the lift force magnitude: L = q_dyn * reference_area * CL
L = q_dyn * reference_area * CL

# Plot the lift force versus angle of attack
plt.figure(figsize=(8, 6))
plt.plot(alpha, L, label='Lift Force')
plt.xlabel('Angle of Attack, α (rad)')
plt.ylabel('Lift Force, L (N)')
plt.title('Lift Force vs. Angle of Attack')
plt.grid(True)
plt.legend()
plt.show()
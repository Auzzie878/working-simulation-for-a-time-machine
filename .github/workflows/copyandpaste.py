import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
c = 3e8  # Speed of light in m/s
B = 30  # Magnetic field strength in Tesla
acceleration = 9.81  # Gravitational acceleration in m/s^2
damping_factor = 0.00001  # Damping factor
mass_particle = 1.67e-27  # Mass of a plasma particle in kg (approx mass of a proton)
num_particles = 1e30  # Number of particles in the plasma
cap_value = 1e24  # Updated cap value to prevent overflow
observer_influence_factor = 1.1  # Factor by which the observer influences the field strength
observer_interval = 45  # Interval at which new observers are introduced in seconds
initial_observers = 1  # Initial number of observers
volume_per_observer = 1.0  # Volume of space added per observer in arbitrary units

# Time parameters
total_time = 9600  # Total simulation
time_steps = 96000  # Number of time steps
dt = total_time / time_steps  # Time step size\

# Arrays to store data
time_array = np.arange(0, total_time, dt)
plasma_velocity_array = np.zeros(len(time_array))
lorentz_factor_array = np.zeros(len(time_array))
energy_density_array = np.zeros(len(time_array))
field_strength_array = np.zeros(len(time_array))
temperature_array = np.zeros(len(time_array))
pressure_array = np.zeros(len(time_array))
quantum_tunneling_probability_array = np.zeros(len(time_array))
spacetime_curvature_array = np.zeros(len(time_array))
volume_array = np.zeros(len(time_array))

# Initial conditions
plasma_velocity = 0.0
lorentz_factor = 1.0
energy_density = 0.0
field_strength = 1e20
temperature = 1e7
pressure = 100000.0  # in Pascals
quantum_tunneling_probability = 0.1
spacetime_curvature = 1e-35
current_observers = initial_observers

# Simulation loop
for i in range(1, len(time_array)):
    try:
        # Update plasma velocity
        plasma_velocity += (acceleration - damping_factor * plasma_velocity) * dt
        lorentz_factor = 1 / np.sqrt(1 - (plasma_velocity / c)**2) if plasma_velocity < c else plasma_velocity / c

        # Apply observer influence at intervals
        if i % observer_interval == 0:
            current_observers *= 2  # Double the number of observers
            if current_observers > 1e6:
                current_observers = 1e6  # Cap the number of observers to prevent overflow

        # Update field strength with observer influence using logarithmic scaling
        observer_factor = np.log1p(observer_influence_factor * current_observers)
        field_strength *= observer_factor

        # Cap field strength to prevent overflow
        if field_strength > cap_value:
            field_strength = cap_value

        # Update energy density with a cap
        energy_density = min(0.5 * (field_strength**2) * (plasma_velocity**2), cap_value)

        # Calculate the volume of space influenced by the observers
        volume = current_observers * volume_per_observer

        # Update temperature
        temperature += 1e5 * dt

        # Update pressure
        pressure += 500.0 * dt

        # Update quantum tunneling probability and spacetime curvature
        quantum_tunneling_probability = 1 - np.exp(-field_strength / cap_value)
        spacetime_curvature = 2 * energy_density / (c**4)

        # Store data
        plasma_velocity_array[i] = plasma_velocity
        lorentz_factor_array[i] = lorentz_factor
        energy_density_array[i] = energy_density
        field_strength_array[i] = field_strength
        temperature_array[i] = temperature
        pressure_array[i] = pressure
        quantum_tunneling_probability_array[i] = quantum_tunneling_probability
        spacetime_curvature_array[i] = spacetime_curvature
        volume_array[i] = volume

    except OverflowError:
        print(f"Overflow error at iteration {i}, capping values")
        plasma_velocity_array[i] = plasma_velocity_array[i-1]
        lorentz_factor_array[i] = lorentz_factor_array[i-1]
        energy_density_array[i] = cap_value
        field_strength_array[i] = field_strength_array[i-1]
        temperature_array[i] = temperature_array[i-1]
        pressure_array[i] = pressure_array[i-1]
        quantum_tunneling_probability_array[i] = quantum_tunneling_probability_array[i-1]
        spacetime_curvature_array[i] = spacetime_curvature_array[i-1]
        volume_array[i] = volume_array[i-1]

# Function to remove non-finite values
def remove_non_finite(data):
    return data[np.isfinite(data)]

# Plotting results
fig, axs = plt.subplots(4, 2, figsize=(15, 10))

axs[0, 0].plot(time_array, plasma_velocity_array)
axs[0, 0].set_title('Plasma Velocity Over Time')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Plasma Velocity (m/s)')

axs[0, 1].plot(time_array, lorentz_factor_array)
axs[0, 1].set_title('Lorentz Factor Over Time')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Lorentz Factor (γ)')

axs[1, 0].plot(time_array, energy_density_array)
axs[1, 0].set_title('Energy Density Over Time')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Energy Density (J/m³)')

axs[1, 1].plot(time_array, field_strength_array)
axs[1, 1].set_title('Field Strength Over Time')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Field Strength (J/m³)')

axs[2, 0].plot(time_array, temperature_array)
axs[2, 0].set_title('Temperature Over Time')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Temperature (K)')

axs[2, 1].plot(time_array, pressure_array)
axs[2, 1].set_title('Pressure Over Time')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Pressure (Pa)')

axs[3, 0].plot(time_array, quantum_tunneling_probability_array)
axs[3, 0].set_title('Quantum Tunneling Probability Over Time')
axs[3, 0].set_xlabel('Time (s)')
axs[3, 0].set_ylabel('Quantum Tunneling Probability')

axs[3, 1].plot(time_array, spacetime_curvature_array)
axs[3, 1].set_title('Spacetime Curvature Over Time')
axs[3, 1].set_xlabel('Time (s)')
axs[3, 1].set_ylabel('Spacetime Curvature (m^-2)')

plt.tight_layout()
plt.show()

# Histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].hist(remove_non_finite(plasma_velocity_array), bins=30, color='blue')
axs[0, 0].set_title('Histogram of Plasma Velocity')
axs[0, 0].set_xlabel('Plasma Velocity (m/s)')

axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(remove_non_finite(lorentz_factor_array), bins=30, color='green')
axs[0, 1].set_title('Histogram of Lorentz Factor')
axs[0, 1].set_xlabel('Lorentz Factor (γ)')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].hist(remove_non_finite(energy_density_array), bins=30, color='red')
axs[1, 0].set_title('Histogram of Energy Density')
axs[1, 0].set_xlabel('Energy Density (J/m³)')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(remove_non_finite(field_strength_array), bins=30, color='purple')
axs[1, 1].set_title('Histogram of Field Strength')
axs[1, 1].set_xlabel('Field Strength (J/m³)')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Scatter Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].scatter(time_array, plasma_velocity_array, color='blue')
axs[0, 0].set_title('Scatter Plot of Plasma Velocity Over Time')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Plasma Velocity (m/s)')

axs[0, 1].scatter(time_array, lorentz_factor_array, color='green')
axs[0, 1].set_title('Scatter Plot of Lorentz Factor Over Time')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Lorentz Factor (γ)')

axs[1, 0].scatter(time_array, energy_density_array, color='red')
axs[1, 0].set_title('Scatter Plot of Energy Density Over Time')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Energy Density (J/m³)')

axs[1, 1].scatter(time_array, field_strength_array, color='purple')
axs[1, 1].set_title('Scatter Plot of Field Strength Over Time')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Field Strength (J/m³)')

plt.tight_layout()
plt.show()

# Box Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].boxplot(remove_non_finite(plasma_velocity_array))
axs[0, 0].set_title('Box Plot of Plasma Velocity')
axs[0, 0].set_ylabel('Plasma Velocity (m/s)')

axs[0, 1].boxplot(remove_non_finite(lorentz_factor_array))
axs[0, 1].set_title('Box Plot of Lorentz Factor')
axs[0, 1].set_ylabel('Lorentz Factor (γ)')

axs[1, 0].boxplot(remove_non_finite(energy_density_array))
axs[1, 0].set_title('Box Plot of Energy Density')
axs[1, 0].set_ylabel('Energy Density (J/m³)')

axs[1, 1].boxplot(remove_non_finite(field_strength_array))
axs[1, 1].set_title('Box Plot of Field Strength')
axs[1, 1].set_ylabel('Field Strength (J/m³)')

plt.tight_layout()
plt.show()

# 3D Histogram for Field Strength Distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

observer_factors = np.arange(1.0, 1.6, 0.1)
X, Y = np.meshgrid(observer_factors, time_array)
Z = np.zeros_like(X)

for j, factor in enumerate(observer_factors):
    current_observers = initial_observers
    field_strength = 1e20
    for i in range(1, len(time_array)):
        if i % observer_interval == 0:
            current_observers *= 1
        observer_factor = np.log1p(observer_influence_factor * current_observers)
        field_strength *= observer_factor
        field_strength = np.log1p(field_strength)
        Z[i, j] = field_strength

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Field Strength Distribution Over Time for Different Observer Influence Factors')
ax.set_xlabel('Observer Influence Factor')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Field Strength (J/m³)')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Constants
c = 3e8  # Speed of light in m/s
B = 40  # Magnetic field strength in Tesla, used as a containment field
initial_acceleration = 2.2  # Initial acceleration in m/s^2
max_acceleration = 13.81  # Maximum gravitational acceleration in m/s^2
damping_factor = 0.00001  # Damping factor
mass_particle = 1.70e-27  # Mass of a plasma particle in kg (approx mass of a proton)
num_particles = 1e45  # Number of particles in the plasma
cap_value = 1e35  # Updated cap value to prevent overflow
observer_influence_factor = 1.1  # Factor by which the observer influences the field strength
observer_interval = 100  # Interval at which new observers are introduced in seconds
initial_observers = 1  # Initial number of observers
volume_per_observer = 1.0  # Volume of space added per observer in arbitrary units

# Time parameters
total_time = 6900  # Total simulation time in seconds
time_steps = 69000  # Number of time steps
dt = total_time / time_steps  # Time step size

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
time_dilation_array = np.zeros(len(time_array))
dimension_portal_probability_array = np.zeros(len(time_array))

# Initial conditions
plasma_velocity = 0.0
lorentz_factor = 1.0
energy_density = 1e-12
field_strength = 1e-20  # Start with a small field strength
temperature = 1e7
pressure = 100000.0  # in Pascals
quantum_tunneling_probability = 0.1
spacetime_curvature = 1e23
current_observers = initial_observers

# Function to gradually increase acceleration
def increasing_acceleration(t, total_time, max_acceleration):
    return initial_acceleration + (max_acceleration - initial_acceleration) * (t / total_time)

# Function to calculate magnetic pressure effect based on magnetic field
def magnetic_pressure_effect(B):
    return B**2 / (2 * mu_0)

# Magnetic permeability of free space
mu_0 = 4 * np.pi * 1e-7

# Simulation loop
for i in range(1, len(time_array)):
    try:
        current_time = time_array[i]

        # Update acceleration
        acceleration = increasing_acceleration(current_time, total_time, max_acceleration)

        # Update plasma velocity
        plasma_velocity += (acceleration - damping_factor * plasma_velocity) * dt
        if plasma_velocity >= c:
            plasma_velocity = c - 1e-10  # Cap to just below the speed of light to avoid division by zero in Lorentz factor
        lorentz_factor = 1 / np.sqrt(1 - (plasma_velocity / c)**2)

        # Apply magnetic pressure effect for stabilization
        pressure = magnetic_pressure_effect(B)
        pressure_array[i] = pressure

        # Gradually introduce observers
        if i % observer_interval == 0 and current_observers < 1e6:
            current_observers += 2  # Slowly increase the number of observers

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

        # Update quantum tunneling probability and spacetime curvature
        quantum_tunneling_probability = 1 - np.exp(-field_strength / cap_value)
        spacetime_curvature = 2 * energy_density / (c**4)

        # Calculate time dilation based on spacetime curvature and Lorentz factor
        time_dilation = lorentz_factor * np.sqrt(1 - (2 * spacetime_curvature * mass_particle) / (c**2))

        # Calculate probability of creating an interdimensional portal
        dimension_portal_probability = np.tanh(spacetime_curvature / cap_value)

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
        time_dilation_array[i] = time_dilation
        dimension_portal_probability_array[i] = dimension_portal_probability

    except OverflowError:
        print(f"Overflow error at iteration {i}, capping values")
        plasma_velocity_array[i] = plasma_velocity_array[i-1]
        lorentz_factor_array[i] = lorentz_factor_array[i-1]
        energy_density_array[i] = cap_value[i-1]
        field_strength_array[i] = field_strength_array[i-1]
        temperature_array[i] = temperature_array[i-1]
        pressure_array[i] = pressure_array[i-1]
        quantum_tunneling_probability_array[i] = quantum_tunneling_probability_array[i-1]
        spacetime_curvature_array[i] = spacetime_curvature_array[i-1]
        volume_array[i] = volume_array[i-1]
        time_dilation_array[i] = time_dilation_array[i-1]
        dimension_portal_probability_array[i] = dimension_portal_probability_array[i-1]

    except ZeroDivisionError:
        print(f"Zero division error at iteration {i}, handling gracefully")
        lorentz_factor = float('inf')
        time_dilation = float('inf')
        plasma_velocity_array[i] = plasma_velocity
        lorentz_factor_array[i] = lorentz_factor
        time_dilation_array[i] = time_dilation

# Function to remove non-finite values
def remove_non_finite(data):
    return data[np.isfinite(data)]

# Save the data to a file
simulation_data = {
    "time_array": time_array.tolist(),
    "plasma_velocity_array": plasma_velocity_array.tolist(),
    "lorentz_factor_array": lorentz_factor_array.tolist(),
    "energy_density_array": energy_density_array.tolist(),
    "field_strength_array": field_strength_array.tolist(),
    "temperature_array": temperature_array.tolist(),
    "pressure_array": pressure_array.tolist(),
    "quantum_tunneling_probability_array": quantum_tunneling_probability_array.tolist(),
    "spacetime_curvature_array": spacetime_curvature_array.tolist(),
    "volume_array": volume_array.tolist(),
    "time_dilation_array": time_dilation_array.tolist(),
    "dimension_portal_probability_array": dimension_portal_probability_array.tolist(),
}

with open('simulation_data.json', 'w') as f:
    json.dump(simulation_data, f)


# Plotting results
fig, axs = plt.subplots(5, 2, figsize=(15, 12))

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

axs[4, 0].plot(time_array, time_dilation_array)
axs[4, 0].set_title('Time Dilation Over Time')
axs[4, 0].set_xlabel('Time (s)')
axs[4, 0].set_ylabel('Time Dilation Factor')

axs[4, 1].plot(time_array, dimension_portal_probability_array)
axs[4, 1].set_title('Dimension Portal Probability Over Time')
axs[4, 1].set_xlabel('Time (s)')
axs[4, 1].set_ylabel('Dimension Portal Probability')

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
    field_strength = 1e-20
    for i in range(1, len(time_array)):
        if i % observer_interval == 0:
            current_observers += 1
        observer_factor = np.log1p(observer_influence_factor * current_observers)
        field_strength *= observer_factor
        field_strength = np.log1p(field_strength)
        Z[i, j] = field_strength

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Field Strength Distribution Over Time for Different Observer Influence Factors')
ax.set_xlabel('Observer Influence Factor')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Field Strength (J/m³)')

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

# Scatter Plots (Phase Space Graphs)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].scatter(plasma_velocity_array, lorentz_factor_array, color='blue')
axs[0, 0].set_title('Phase Space: Plasma Velocity vs Lorentz Factor')
axs[0, 0].set_xlabel('Plasma Velocity (m/s)')
axs[0, 0].set_ylabel('Lorentz Factor (γ)')

axs[0, 1].scatter(energy_density_array, field_strength_array, color='green')
axs[0, 1].set_title('Phase Space: Energy Density vs Field Strength')
axs[0, 1].set_xlabel('Energy Density (J/m³)')
axs[0, 1].set_ylabel('Field Strength (J/m³)')

axs[1, 0].scatter(temperature_array, pressure_array, color='red')
axs[1, 0].set_title('Phase Space: Temperature vs Pressure')
axs[1, 0].set_xlabel('Temperature (K)')
axs[1, 0].set_ylabel('Pressure (Pa)')

axs[1, 1].scatter(spacetime_curvature_array, dimension_portal_probability_array, color='purple')
axs[1, 1].set_title('Phase Space: Spacetime Curvature vs Dimension Portal Probability')
axs[1, 1].set_xlabel('Spacetime Curvature (m^-2)')
axs[1, 1].set_ylabel('Dimension Portal Probability')

plt.tight_layout()
plt.show()

# Technical Analysis
average_dimension_portal_probability = np.mean(remove_non_finite(dimension_portal_probability_array))
max_dimension_portal_probability = np.max(remove_non_finite(dimension_portal_probability_array))

print(f"Average Probability of Creating a Dimension Portal: {average_dimension_portal_probability:.6f}")
print(f"Maximum Probability of Creating a Dimension Portal: {max_dimension_portal_probability:.6f}")

import csv

# Define the CSV file path
csv_file_path = 'simulation_results.csv'

# Define the header
header = [
    'Time (s)', 'Plasma Velocity (m/s)', 'Lorentz Factor (γ)', 'Energy Density (J/m³)',
    'Field Strength (J/m³)', 'Temperature (K)', 'Pressure (Pa)',
    'Quantum Tunneling Probability', 'Spacetime Curvature (m^-2)',
    'Time Dilation Factor', 'Dimension Portal Probability'
]

# Write data to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(header)

    # Write the data
    for i in range(len(time_array)):
        writer.writerow([
            time_array[i], plasma_velocity_array[i], lorentz_factor_array[i],
            energy_density_array[i], field_strength_array[i], temperature_array[i],
            pressure_array[i], quantum_tunneling_probability_array[i],
            spacetime_curvature_array[i], time_dilation_array[i],
            dimension_portal_probability_array[i]
        ])

print(f"Simulation results saved to {csv_file_path}")

import pandas as pd

# Assuming the arrays are already created and populated in your simulation
simulation_data = {
    "time_array": time_array,
    "plasma_velocity_array": plasma_velocity_array,
    "lorentz_factor_array": lorentz_factor_array,
    "energy_density_array": energy_density_array,
    "field_strength_array": field_strength_array,
    "temperature_array": temperature_array,
    "pressure_array": pressure_array,
    "quantum_tunneling_probability_array": quantum_tunneling_probability_array,
    "spacetime_curvature_array": spacetime_curvature_array,
    "time_dilation_array": time_dilation_array,
    "dimension_portal_probability_array": dimension_portal_probability_array,
}

# Create a DataFrame
df = pd.DataFrame(simulation_data)

# Save the DataFrame to a CSV file
df.to_csv("simulation_data.csv", index=False)

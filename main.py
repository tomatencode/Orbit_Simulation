import numpy as np
import numpy.typing as npt
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
r_earth = 6731000  # [m]
m_earth = 5.972e24 # [kg]
T_0 = 288.15       # Sea level standard temperature [K]
P_0 = 101325       # Sea level standard pressure [Pa]
L_tropo = 0.0065   # Temperature lapse rate in the troposphere [K/m]
R = const.R     # Universal gas constant [J/(mol*K)]
M_air = 0.0289644  # Molar mass of Earth's air [kg/mol]
G = const.G        # gravity const
g = const.g        # gravity const

# Define temperature lapse rates and constants for other layers
L_strato_lower = 0       # Stratosphere lower temperature gradient [K/m]
L_strato_upper = 0.001   # Stratosphere upper temperature gradient [K/m]
L_meso = 0.0028          # Mesosphere temperature gradient [K/m]

# Function to calculate pressure at a given altitude using the barometric formula
def pressure_at_altitude(P_b, T_b, L_b, h, h_b):
    if L_b == 0:  # Isothermal layer
        return P_b * np.exp(-g * M_air * (h - h_b) / (R * T_b))
    else:  # Gradient layer
        return P_b * (1 + L_b * (h - h_b) / T_b) ** (-g * M_air / (R * L_b))

# Function to calculate temperature at a given altitude
def temperature_at_altitude(T_b, L_b, h, h_b):
    return T_b + L_b * (h - h_b)

# Function to calculate density at a given altitude
def density_at_altitude(altitude):
    if altitude <= 11000:  # Troposphere
        T = temperature_at_altitude(T_0, -L_tropo, altitude, 0)
        P = pressure_at_altitude(P_0, T_0, -L_tropo, altitude, 0)
    elif altitude <= 20000:  # Lower Stratosphere
        T = temperature_at_altitude(T_0 - L_tropo * 11000, L_strato_lower, altitude, 11000)
        P = pressure_at_altitude(P_0 * (1 - L_tropo * 11000 / T_0) ** (g * M_air / (R * L_tropo)), T_0 - L_tropo * 11000, L_strato_lower, altitude, 11000)
    elif altitude <= 32000:  # Upper Stratosphere
        T = temperature_at_altitude(T_0 - L_tropo * 11000, L_strato_upper, altitude, 20000)
        P = pressure_at_altitude(P_0 * (1 - L_tropo * 11000 / T_0) ** (g * M_air / (R * L_tropo)) * np.exp(-g * M_air * (20000 - 11000) / (R * (T_0 - L_tropo * 11000))), T_0 - L_tropo * 11000, L_strato_upper, altitude, 20000)
    elif altitude <= 47000:  # Mesosphere
        T = temperature_at_altitude(T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000), L_meso, altitude, 32000)
        P = pressure_at_altitude(P_0 * (1 - L_tropo * 11000 / T_0) ** (g * M_air / (R * L_tropo)) * np.exp(-g * M_air * (32000 - 11000) / (R * (T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000)))), T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000), L_meso, altitude, 32000)
    elif altitude <= 85000:  # Mesosphere
        T = temperature_at_altitude(T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000) - L_meso * (47000 - 32000), 0, altitude, 47000)
        P = pressure_at_altitude(P_0 * (1 - L_tropo * 11000 / T_0) ** (g * M_air / (R * L_tropo)) * np.exp(-g * M_air * (47000 - 11000) / (R * (T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000)))), T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000) - L_meso * (47000 - 32000), 0, altitude, 47000)
    else:  # Thermosphere
        T = T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000) - L_meso * (47000 - 32000)
        P = pressure_at_altitude(P_0 * (1 - L_tropo * 11000 / T_0) ** (g * M_air / (R * L_tropo)) * np.exp(-g * M_air * (85000 - 11000) / (R * (T_0 - L_tropo * 11000 + L_strato_upper * (32000 - 20000)))), T, 0, altitude, 85000)
    
    rho_air = P / (R / M_air * T)
    return rho_air


class Rocket:
    def __init__(self, C_drag, A_top, start_alt, m) -> None:
        self.C_drag = C_drag # []
        self.A_top = A_top # [m²]
        self.m = m # [kg]
        self.flight_time = 0 # [s]
        self.stage = 1
        
        self.pos = np.array([0,start_alt + r_earth], dtype=np.float64) # [m]
        self.vel = np.array([465,0], dtype=np.float64) # [m/s] (starts at 465 because of earths rorotation)
    
    
    def get_f_drag(self) -> npt.NDArray[np.float64]:
        alt = np.linalg.norm(self.pos)
        
        v_vel_lenght = np.linalg.norm(self.vel)
        if v_vel_lenght != 0.0:
            vel_norm = self.vel / v_vel_lenght
        else:
            vel_norm = np.array([0.0,0.0])
        
        drag = 0.5 * C_drag * density_at_altitude(alt - r_earth) * v_vel_lenght**2 * A_top
        v_drag = -vel_norm * drag
        return v_drag

    def get_f_gravitiy(self) -> npt.NDArray[np.float64]:
        distance_from_center = np.linalg.norm(self.pos)

        pos_norm = self.pos / distance_from_center
        
        # Calculate gravitational force
        f_gravity = -pos_norm * G * (m_earth * self.m) / (distance_from_center ** 2)
        
        return f_gravity
    
    def get_f_motor(self):
        if self.flight_time < 8:
            alt = np.linalg.norm(self.pos)
            
            desired_angle = -45
            
            pos_norm = self.pos / alt
            cos_pitch = np.cos(np.radians(desired_angle))
            sin_pitch = np.sin(np.radians(desired_angle))

            direction_norm = np.array([
                cos_pitch * pos_norm[0] - sin_pitch * pos_norm[1],
                sin_pitch * pos_norm[0] + cos_pitch * pos_norm[1]
            ])
            
            return  direction_norm * 500
        
        elif 250 < self.flight_time < 260:
            self.stage = 2
            alt = np.linalg.norm(self.pos)
            
            desired_angle = -95
            
            pos_norm = self.pos / alt
            cos_pitch = np.cos(np.radians(desired_angle))
            sin_pitch = np.sin(np.radians(desired_angle))

            direction_norm = np.array([
                cos_pitch * pos_norm[0] - sin_pitch * pos_norm[1],
                sin_pitch * pos_norm[0] + cos_pitch * pos_norm[1]
            ])
            
            return  direction_norm * 504
        
        return np.array([0, 0])
    
    def update(self, dt):
        if np.linalg.norm(self.pos) > r_earth:
            self.vel += (self.get_f_gravitiy() + self.get_f_drag() + self.get_f_motor())/m * dt
            self.pos += self.vel * dt
            self.flight_time += dt

# optimistic rocet parameters
start_alt = 36000
C_drag = 0.75
radius = 0.04
A_top = 2 * np.pi * radius**2
m = 1.0

rocket = Rocket(C_drag,A_top,start_alt,m)

fig, ax = plt.subplots(figsize=(10, 10))
circle = plt.Circle((0, 0), r_earth/1000, edgecolor='b', facecolor='none')
# Add the circle to the plot
ax.add_patch(circle)

line, = ax.plot([], [])  # Create an empty plot

# zoomd in
#ax.set_ylim((r_earth/1000 - 10), (r_earth/1000 + 200))
#ax.set_xlim(-210, 210)

# zoomd out
ax.set_ylim(-(r_earth/1000 + 6000), (r_earth/1000 + 6000))
ax.set_xlim(-(r_earth/1000 + 6000), (r_earth/1000 + 6000))

ax.set_title("Orbit")

steps_per_frame = 100

x = []
y = []

def animate(frame):
    for _ in range(steps_per_frame):
        rocket.update(0.1)
    
    x.append(rocket.pos[0] / 1000)
    y.append(rocket.pos[1] / 1000)
    
    line.set_data(x, y)
    return line,

# Create infinite animation
anim = animation.FuncAnimation(fig, animate, frames=None, interval=1, blit=True, cache_frame_data=False)

# Number of frames
#frames = 700

# Create animation
#anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1)

# Save animation as gif
#anim.save('orbit.gif', writer='pillow', fps=30)
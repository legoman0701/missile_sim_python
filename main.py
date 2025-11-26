import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Interactive3DPath:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Physics parameters (SI units: meters, seconds, kg)
        self.bounds = 10000.0  # Cube boundary (10 km = 10000 m)
        self.gravity = np.array([0.0, 0.0, -9.81])  # Gravity (m/s²)
        self.damping = 0.95  # Energy loss on collision
        self.dot_radius = 50.0  # Collision radius (50 m)
        self.dt = 0.1  # Time step (seconds)
        
        # Initialize positions (meters)
        self.blue_pos = np.array([5000.0, 0.0, 5000.0])  # Start on circle (5km)
        self.red_pos = np.array([0.0, 0.0, 0.0])  # Start at origin (0,0,0)
        
        # Initialize velocities (m/s)
        self.blue_vel = np.array([0.0, 0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0, 0.0])
        
        # Path history
        self.blue_path = [self.blue_pos.copy()]
        self.red_path = [self.red_pos.copy()]
        
        # Physics state
        self.physics_enabled = True
        
        # Blue dot circular flight parameters
        self.blue_auto_fly = True
        self.flight_altitude = 5000.0  # 5km altitude (meters)
        self.flight_radius = 5000.0    # 5km radius (meters)
        
        # Speed: 700 km/h = 194.44 m/s
        # Angular speed (rad/s) = linear_speed / radius = 194.44 / 5000 = 0.03889 rad/s
        linear_speed = 700.0 * 1000.0 / 3600.0  # 700 km/h to m/s
        self.flight_angular_speed = linear_speed / self.flight_radius  # rad/s
        self.flight_angle = 0.0  # Current angle (radians)
        
        # Plot elements
        self.blue_dot, = self.ax.plot([self.blue_pos[0]/1000.0], [self.blue_pos[1]/1000.0], [self.blue_pos[2]/1000.0], 
                                       'bo', markersize=10, label='Blue (700 km/h)')
        self.red_dot, = self.ax.plot([self.red_pos[0]/1000.0], [self.red_pos[1]/1000.0], [self.red_pos[2]/1000.0], 
                                      'ro', markersize=10, label='Red (Physics)')
        
        self.blue_line, = self.ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6)
        self.red_line, = self.ax.plot([], [], [], 'r-', linewidth=2, alpha=0.6)
        
        # Setup plot (display in kilometers for readability)
        bounds_km = self.bounds / 1000.0
        self.ax.set_xlim(-bounds_km, bounds_km)
        self.ax.set_ylim(-bounds_km, bounds_km)
        self.ax.set_zlim(-bounds_km, bounds_km)
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.legend()
        self.ax.set_title('3D Flight Tracker - Blue: 700 km/h circular, Red: Physics')
        
        # Draw boundary cube
        self.draw_boundary_cube()
        
        # Current selection
        self.selected = 'red'
        
        # Add keyboard support
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Start physics animation (100ms interval = 0.1s = dt)
        self.animation_timer = self.fig.canvas.new_timer(interval=int(self.dt * 1000))
        self.animation_timer.add_callback(self.update_physics)
        self.animation_timer.start()
    
    def draw_boundary_cube(self):
        """Draw the boundary cube (in km for display)"""
        b = self.bounds / 1000.0  # Convert to km
        edges = [
            # Bottom square
            [[-b, -b, -b], [b, -b, -b]], [[b, -b, -b], [b, b, -b]],
            [[b, b, -b], [-b, b, -b]], [[-b, b, -b], [-b, -b, -b]],
            # Top square
            [[-b, -b, b], [b, -b, b]], [[b, -b, b], [b, b, b]],
            [[b, b, b], [-b, b, b]], [[-b, b, b], [-b, -b, b]],
            # Vertical edges
            [[-b, -b, -b], [-b, -b, b]], [[b, -b, -b], [b, -b, b]],
            [[b, b, -b], [b, b, b]], [[-b, b, -b], [-b, b, b]]
        ]
        
        for edge in edges:
            points = np.array(edge)
            self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.3, linewidth=0.5)
    
    def toggle_physics(self):
        """Toggle physics simulation on/off"""
        self.physics_enabled = not self.physics_enabled
        status = "ON" if self.physics_enabled else "OFF"
        print(f"Physics: {status}")
    
    def toggle_autofly(self):
        """Toggle blue dot auto-fly on/off"""
        self.blue_auto_fly = not self.blue_auto_fly
        status = "ON" if self.blue_auto_fly else "OFF"
        print(f"Blue Auto-Fly: {status}")
    
    def check_boundary_collision(self, pos, vel):
        """Check and handle collisions with boundary cube (SI units)"""
        for i in range(3):
            if pos[i] - self.dot_radius < -self.bounds:
                pos[i] = -self.bounds + self.dot_radius
                vel[i] = abs(vel[i]) * self.damping
            elif pos[i] + self.dot_radius > self.bounds:
                pos[i] = self.bounds - self.dot_radius
                vel[i] = -abs(vel[i]) * self.damping
        return pos, vel
    
    def check_dot_collision(self):
        """Check and handle collision between the two dots (SI units)"""
        diff = self.blue_pos - self.red_pos
        dist = np.linalg.norm(diff)
        min_dist = 2 * self.dot_radius
        
        if dist < min_dist and dist > 0:
            # Collision detected
            normal = diff / dist
            
            # Relative velocity
            rel_vel = self.blue_vel - self.red_vel
            
            # Velocity component along collision normal
            vel_along_normal = np.dot(rel_vel, normal)
            
            # Only resolve if dots are moving toward each other
            if vel_along_normal < 0:
                # Elastic collision with damping
                impulse = vel_along_normal * normal * self.damping
                self.blue_vel -= impulse
                self.red_vel += impulse
                
                # Separate the dots
                overlap = min_dist - dist
                separation = normal * overlap / 2
                self.blue_pos += separation
                self.red_pos -= separation
    
    def update_physics(self):
        """Update physics simulation using proper SI units and dt"""
        # Handle blue dot circular flight
        if self.blue_auto_fly:
            # Update angle: dθ = ω * dt
            self.flight_angle += self.flight_angular_speed * self.dt
            
            # Calculate position on circle (meters)
            self.blue_pos[0] = self.flight_radius * np.cos(self.flight_angle)
            self.blue_pos[1] = self.flight_radius * np.sin(self.flight_angle)
            self.blue_pos[2] = self.flight_altitude
            
            # Calculate velocity (tangent to circle, m/s)
            self.blue_vel[0] = -self.flight_radius * self.flight_angular_speed * np.sin(self.flight_angle)
            self.blue_vel[1] = self.flight_radius * self.flight_angular_speed * np.cos(self.flight_angle)
            self.blue_vel[2] = 0.0
        elif self.physics_enabled:
            # Apply gravity: v = v + a*dt
            self.blue_vel += self.gravity * self.dt
            # Update position: x = x + v*dt
            self.blue_pos += self.blue_vel * self.dt
            # Check boundary collisions
            self.blue_pos, self.blue_vel = self.check_boundary_collision(self.blue_pos, self.blue_vel)
        
        # Red dot always follows physics
        if self.physics_enabled:
            # Apply gravity: v = v + a*dt
            self.red_vel += self.gravity * self.dt
            # Update position: x = x + v*dt
            self.red_pos += self.red_vel * self.dt
            # Check boundary collisions
            self.red_pos, self.red_vel = self.check_boundary_collision(self.red_pos, self.red_vel)
        
        # Check dot collision
        if self.physics_enabled:
            self.check_dot_collision()
        
        # Update path history
        if self.blue_auto_fly or np.linalg.norm(self.blue_vel) > 0.1:
            self.blue_path.append(self.blue_pos.copy())
            if len(self.blue_path) > 1000:
                self.blue_path.pop(0)
        
        if np.linalg.norm(self.red_vel) > 0.1:
            self.red_path.append(self.red_pos.copy())
            if len(self.red_path) > 1000:
                self.red_path.pop(0)
        
        # Update plot (convert to km for display)
        self.blue_dot.set_data([self.blue_pos[0]/1000.0], [self.blue_pos[1]/1000.0])
        self.blue_dot.set_3d_properties([self.blue_pos[2]/1000.0])
        
        self.red_dot.set_data([self.red_pos[0]/1000.0], [self.red_pos[1]/1000.0])
        self.red_dot.set_3d_properties([self.red_pos[2]/1000.0])
        
        # Update paths
        if len(self.blue_path) > 1:
            path_array = np.array(self.blue_path) / 1000.0  # Convert to km
            self.blue_line.set_data(path_array[:, 0], path_array[:, 1])
            self.blue_line.set_3d_properties(path_array[:, 2])
        
        if len(self.red_path) > 1:
            path_array = np.array(self.red_path) / 1000.0  # Convert to km
            self.red_line.set_data(path_array[:, 0], path_array[:, 1])
            self.red_line.set_3d_properties(path_array[:, 2])
        
        self.fig.canvas.draw_idle()
        
    def select_dot(self, color):
        """Select which dot to control"""
        self.selected = color
        print(f"Selected: {color.upper()}")
        
    def move_dot(self, axis, direction):
        """Apply velocity impulse to selected dot (m/s)"""
        impulse = 50.0  # m/s impulse
        
        if self.selected == 'blue':
            # Disable auto-fly if manually controlling blue dot
            if self.blue_auto_fly:
                self.blue_auto_fly = False
                print("Blue Auto-Fly: OFF (manual control)")
            self.blue_vel[axis] += direction * impulse
        else:
            self.red_vel[axis] += direction * impulse
        
    def clear_path(self):
        """Clear path history for selected dot"""
        if self.selected == 'blue':
            self.blue_path = [self.blue_pos.copy()]
            self.blue_line.set_data([], [])
            self.blue_line.set_3d_properties([])
        else:
            self.red_path = [self.red_pos.copy()]
            self.red_line.set_data([], [])
            self.red_line.set_3d_properties([])
        
        self.fig.canvas.draw_idle()
        print(f"Cleared {self.selected} path")
        
    def on_key(self, event):
        """Keyboard controls"""
        if event.key == 'b':
            self.select_dot('blue')
        elif event.key == 'r':
            self.select_dot('red')
        elif event.key == 'right':
            self.move_dot(0, 1)  # X+
        elif event.key == 'left':
            self.move_dot(0, -1)  # X-
        elif event.key == 'up':
            self.move_dot(1, 1)  # Y+
        elif event.key == 'down':
            self.move_dot(1, -1)  # Y-
        elif event.key == 'pageup':
            self.move_dot(2, 1)  # Z+
        elif event.key == 'pagedown':
            self.move_dot(2, -1)  # Z-
        elif event.key == 'c':
            self.clear_path()
        elif event.key == 'p':
            self.toggle_physics()
        elif event.key == 'f':
            self.toggle_autofly()
    
    def show(self):
        print("3D Flight Tracker (SI Units)")
        print("=" * 40)
        print("Blue Dot: 700 km/h circular flight at 5km altitude")
        print("Red Dot: Physics simulation with gravity")
        print("\nKeyboard Controls:")
        print("  b/r     - Select blue/red dot")
        print("  Arrows  - Move X/Y axes")
        print("  PgUp/Dn - Move Z axis")
        print("  c       - Clear selected path")
        print("  p       - Toggle physics")
        print("  f       - Toggle blue auto-fly")
        print("\nPhysics Parameters:")
        print(f"  Boundary: ±{self.bounds/1000:.0f} km")
        print(f"  Gravity: {abs(self.gravity[2]):.2f} m/s²")
        print(f"  Time step: {self.dt} s")
        plt.show()

if __name__ == "__main__":
    tracker = Interactive3DPath()
    tracker.show()

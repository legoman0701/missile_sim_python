import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for better performance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Performance optimizations for matplotlib
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

class MissileSimulation:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('Missile Simulation - Performance Mode')
        
        # Single plot for maximum performance
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Disable auto-scaling for performance
        self.ax.set_autoscale_on(False)
        
        # Disable closeup view for performance
        self.closeup_enabled = False
        
        # Physics parameters (SI units)
        self.bounds = 10000.0  # Cube boundary (10 km)
        self.gravity = np.array([0.0, 0.0, -9.81])  # Gravity (m/s²)
        self.dt = 0.05  # Time step (20 FPS - optimal for matplotlib 3D)
        
        # Target (aircraft flying in circle)
        self.target_pos = np.array([5000.0, 0.0, 5000.0])
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.target_altitude = 5000.0  # 5km
        self.target_radius = 5000.0    # 5km radius
        target_speed = 700.0 * 1000.0 / 3600.0  # 700 km/h to m/s
        self.target_angular_speed = target_speed / self.target_radius
        self.target_angle = 0.0
        
        # Missile parameters
        self.missile_pos = np.array([0.0, 0.0, 50.0])  # Spawn at launch tower
        self.missile_vel = np.array([0.0, 0.0, 0.0])
        self.missile_spawned = True  # Missile is visible
        self.missile_active = False  # Missile is flying
        self.missile_thrust = 150.0  # m/s² acceleration (typical SAM: 50-150 m/s²)
        self.missile_mass = 85.0  # kg (similar to AIM-9)
        self.missile_fuel = 20.0  # seconds of fuel
        self.missile_time = 0.0
        
        # Aerodynamic parameters (AIM-9 style)
        self.body_drag_coeff = 0.3  # Body drag coefficient
        self.body_area = 0.02  # m² (cross-sectional area)
        
        # Control surfaces - 4 individual Canards (front)
        self.canard_area = 0.03  # m² each canard
        # Deflection for each canard: [top, bottom, left, right] in radians
        self.canard_deflection = np.array([0.0, 0.0, 0.0, 0.0])
        self.canard_max_deflection = np.radians(25)  # 25 degrees max
        self.canard_distance = 0.5  # meters from CoG
        self.canard_lift_slope = 5.0  # lift coefficient per radian (Cl_alpha)
        self.canard_stall_angle = np.radians(15)  # 15 degrees stall
        
        # 4 individual Stabilizing fins (rear) - can be controlled or passive
        self.fin_area = 0.04  # m² each fin
        # Deflection for each fin: [top, bottom, left, right] in radians
        self.fin_deflection = np.array([0.0, 0.0, 0.0, 0.0])
        self.fin_max_deflection = np.radians(15)  # Smaller range for stability
        self.fin_distance = 1.5  # meters from CoG
        self.fin_damping = 0.8  # Stabilization damping coefficient
        self.fin_lift_slope = 3.5  # Less than canards
        
        # Rollerons (roll control surfaces) - 2 small surfaces on rear fins
        self.rolleron_area = 0.015  # m² each rolleron (small)
        self.rolleron_distance = 1.8  # meters from CoG (at fin tips)
        self.rolleron_damping = 1.5  # Roll damping coefficient
        self.rolleron_lift_slope = 2.0  # Lift coefficient per radian
        
        # Missile orientation (quaternion or Euler angles - using body frame)
        # Initial orientation pointing straight up (pitch = -pi/2 yields forward +Z after axis fix)
        self.missile_pitch = -np.pi / 2.0
        self.missile_yaw = 0.0  # No horizontal rotation
        self.missile_roll = 0.0  # No roll
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # rad/s [pitch_rate, yaw_rate, roll_rate]
        self.moment_of_inertia = 2.0  # kg*m² simplified scalar
        
        # Guidance system
        self.guidance_gain = 3  # Proportional navigation constant (lower = less lead)
        
        # Simulation speed control
        self.sim_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
        
        # Path history
        self.target_path = [self.target_pos.copy()]
        self.missile_path = [self.missile_pos.copy()]
        
        # Plot elements (with antialiasing disabled for performance)
        self.target_dot, = self.ax.plot([self.target_pos[0]/1000.0], 
                                        [self.target_pos[1]/1000.0], 
                                        [self.target_pos[2]/1000.0], 
                                        'bo', markersize=10, label='Target (700 km/h)', 
                                        antialiased=False)
        
        self.missile_dot, = self.ax.plot([], [], [], 'ro', markersize=8, label='Missile', 
                                         antialiased=False)
        self.target_line, = self.ax.plot([], [], [], 'b-', linewidth=1.5, alpha=0.5, 
                                         antialiased=False)
        self.missile_line, = self.ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7, 
                                          antialiased=False)
        
        # Debug visualization lines (antialiasing disabled)
        self.los_line, = self.ax.plot([], [], [], 'g--', linewidth=2, alpha=0.7, 
                                       label='Line of Sight', antialiased=False)
        self.missile_vel_line, = self.ax.plot([], [], [], 'r-', linewidth=2, alpha=0.8, 
                                               antialiased=False)
        self.target_vel_line, = self.ax.plot([], [], [], 'b-', linewidth=2, alpha=0.8, 
                                              antialiased=False)
        self.guidance_vec_line, = self.ax.plot([], [], [], 'y-', linewidth=3, alpha=0.9, 
                                                label='Guidance', antialiased=False)
        
        # Setup main plot
        bounds_km = self.bounds / 1000.0
        self.ax.set_xlim(-bounds_km, bounds_km)
        self.ax.set_ylim(-bounds_km, bounds_km)
        self.ax.set_zlim(0, bounds_km * 2)
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.legend(loc='upper left', fontsize=8)
        self.ax.set_title('Main View - Press SPACE to launch')
        
        # Reduce grid density for performance
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        self.ax.zaxis.set_major_locator(plt.MaxNLocator(5))
        
        # Closeup view disabled for performance
        
        # Draw boundary cube
        self.draw_boundary_cube()
        
        # Text display for info
        self.info_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes, 
                                       verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        # Closeup view completely disabled for maximum performance
        # (OBJ loading and 3D mesh rendering is too expensive for matplotlib)
        
        # Keyboard controls
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Animation timer (60 FPS)
        self.animation_timer = self.fig.canvas.new_timer(interval=int(self.dt * 1000))
        self.animation_timer.add_callback(self.update)
        self.animation_timer.start()
        
        # Cache for rotation matrix
        self._cached_rotation_matrix = None
        self._cached_orientation = None
        # Performance helpers
        self.frame_count = 0
        self.closeup_update_interval = 10  # update closeup every N frames (drastically reduced)
        self.path_update_interval = 2  # only update paths every N frames
        
        # Rendering optimization
        self.use_blit = True
        self.background = None
        self.skip_frames = 0  # Frame skipping counter
    
    def load_obj_model(self, filepath):
        """Load OBJ file and parse mesh data by object name"""
        import os
        obj_path = os.path.join(os.path.dirname(__file__), filepath)
        
        vertices = []
        objects = {}
        current_obj = None
        
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):  # Vertex
                    parts = line.split()
                    # Convert to [x, y, z] - Blender Z-up to our coordinate system
                    # Swap axes: Blender X,Y,Z -> our Z,X,Y (missile length along X)
                    vertices.append([float(parts[3]), float(parts[1]), float(parts[2])])
                elif line.startswith('o '):  # Object name
                    current_obj = line.split()[1]
                    objects[current_obj] = []
                elif line.startswith('f ') and current_obj:  # Face
                    parts = line.split()[1:]
                    # Parse face indices (format: v/vt/vn or v//vn or v)
                    face_verts = []
                    for part in parts:
                        idx = int(part.split('/')[0]) - 1  # OBJ indices are 1-based
                        face_verts.append(vertices[idx])
                    objects[current_obj].append(face_verts)
        
        # Calculate center of each object for rotation pivots
        obj_centers = {}
        for obj_name, faces in objects.items():
            all_verts = []
            for face in faces:
                all_verts.extend(face)
            if all_verts:
                center = np.mean(all_verts, axis=0)
                obj_centers[obj_name] = center
        
        self.obj_centers = obj_centers
        return objects
    
    def draw_boundary_cube(self):
        """Draw boundary cube (in km)"""
        b = self.bounds / 1000.0
        edges = [
            [[-b, -b, 0], [b, -b, 0]], [[b, -b, 0], [b, b, 0]],
            [[b, b, 0], [-b, b, 0]], [[-b, b, 0], [-b, -b, 0]],
            [[-b, -b, b*2], [b, -b, b*2]], [[b, -b, b*2], [b, b, b*2]],
            [[b, b, b*2], [-b, b, b*2]], [[-b, b, b*2], [-b, -b, b*2]],
            [[-b, -b, 0], [-b, -b, b*2]], [[b, -b, 0], [b, -b, b*2]],
            [[b, b, 0], [b, b, b*2]], [[-b, b, 0], [-b, b, b*2]]
        ]
        
        for edge in edges:
            points = np.array(edge)
            self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.2, linewidth=0.5)
    
    def launch_missile(self):
        """Launch missile - start flight"""
        if not self.missile_active and self.missile_spawned:
            self.missile_vel = np.array([0.0, 0.0, 100.0])  # Initial upward velocity
            self.angular_velocity = np.array([0.0, 0.0, 0.0])
            self.missile_active = True
            self.missile_time = 0.0
            print(f"MISSILE LAUNCHED! (Sim speed: {self.sim_speed:.1f}x)")
    
    def get_body_axes(self):
        """Get missile body axes based on current orientation"""
        # Forward (x-axis in body frame)
        # Define forward so pitch = -pi/2 points along +Z (upward)
        forward = np.array([
            np.cos(self.missile_yaw) * np.cos(self.missile_pitch),
            np.sin(self.missile_yaw) * np.cos(self.missile_pitch),
            -np.sin(self.missile_pitch)
        ])
        
        # Right (y-axis in body frame)
        right = np.array([
            -np.sin(self.missile_yaw),
            np.cos(self.missile_yaw),
            0.0
        ])
        
        # Up (z-axis in body frame)
        up = np.cross(forward, right)
        
        return forward, right, up
    
    def calculate_aero_forces(self, speed):
        """Calculate aerodynamic forces from control surfaces with realistic aerodynamics"""
        if speed < 1.0:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        
        forward, right, up = self.get_body_axes()
        vel_dir = self.missile_vel / speed
        
        # Air density (kg/m³) - decreases with altitude
        altitude = self.missile_pos[2]
        rho = 1.225 * np.exp(-altitude / 8500.0)  # Exponential atmosphere model
        q = 0.5 * rho * speed * speed  # Dynamic pressure
        
        total_force = np.array([0.0, 0.0, 0.0])
        total_moment = np.array([0.0, 0.0, 0.0])
        
        # Body drag (always opposes velocity)
        drag_force = -vel_dir * q * self.body_drag_coeff * self.body_area
        total_force += drag_force
        
        # CANARDS: [top, bottom, left, right]
        self.canard_forces = []
        canard_positions = [
            ('Can_TOP', 0),
            ('Can_DOWN', 1),
            ('Can_LEFT', 2),
            ('Can_RIGHT', 3)
        ]
        
        for obj_name, idx in canard_positions:
            if obj_name not in self.obj_centers:
                self.canard_forces.append(np.array([0.0, 0.0, 0.0]))
                continue
            
            deflection = self.canard_deflection[idx]
            
            # Get surface center position in body frame
            surface_center = np.array(self.obj_centers[obj_name])
            
            # Calculate surface normal based on deflection
            if idx < 2:  # Top/Bottom - rotate around Z axis
                initial_normal = np.array([0, 1, 0]) if idx == 0 else np.array([0, -1, 0])
                cos_a, sin_a = np.cos(deflection), np.sin(deflection)
                rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                surface_normal = rot_z @ initial_normal
            else:  # Left/Right - rotate around Y axis
                initial_normal = np.array([0, 0, 1]) if idx == 2 else np.array([0, 0, -1])
                cos_a, sin_a = np.cos(deflection), np.sin(deflection)
                rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
                surface_normal = rot_y @ initial_normal
            
            # Convert surface normal to world frame
            R = self.get_rotation_matrix()
            surface_normal_world = R @ surface_normal
            
            # Calculate angle of attack (angle between velocity and surface normal)
            # The lift is perpendicular to both the velocity and the surface
            cos_aoa = np.dot(vel_dir, surface_normal_world)
            sin_aoa = np.sqrt(max(0, 1 - cos_aoa**2))
            aoa = np.arcsin(sin_aoa) * np.sign(cos_aoa)
            
            # Calculate lift coefficient with stall
            if abs(aoa) < self.canard_stall_angle:
                Cl = self.canard_lift_slope * aoa
            else:
                Cl = self.canard_lift_slope * self.canard_stall_angle * np.sign(aoa) * 0.5
            
            # Lift direction: perpendicular to velocity, in the plane of velocity and normal
            # Lift = q * Cl * A * lift_direction
            if sin_aoa > 0.01:  # Avoid division by zero
                lift_dir = surface_normal_world - vel_dir * cos_aoa
                lift_dir = lift_dir / np.linalg.norm(lift_dir)
            else:
                lift_dir = surface_normal_world
            
            lift_magnitude = q * Cl * self.canard_area
            lift_force = lift_dir * lift_magnitude
            
            # Drag on the surface (proportional to sin²(aoa))
            surface_drag = -vel_dir * q * 0.1 * self.canard_area * (sin_aoa**2)
            
            total_surface_force = lift_force + surface_drag
            total_force += total_surface_force
            self.canard_forces.append(total_surface_force)
            
            # Calculate moment around CoG
            # Position of force application (surface center) in world frame
            force_position_world = R @ surface_center
            moment_arm = force_position_world
            total_moment += np.cross(moment_arm, total_surface_force)
        
        # REAR FINS: [top, bottom, left, right]
        self.fin_forces = []
        fin_positions = [
            ('Stab_TOP', 0),
            ('Stab_DOWN', 1),
            ('Stab_LEFT', 2),
            ('Stab_RIGHT', 3)
        ]
        
        for obj_name, idx in fin_positions:
            if obj_name not in self.obj_centers:
                self.fin_forces.append(np.array([0.0, 0.0, 0.0]))
                continue
            
            deflection = self.fin_deflection[idx]
            
            # Get surface center position in body frame
            surface_center = np.array(self.obj_centers[obj_name])
            
            # Calculate surface normal based on deflection
            if idx < 2:  # Top/Bottom - rotate around Z axis
                initial_normal = np.array([0, 1, 0]) if idx == 0 else np.array([0, -1, 0])
                cos_a, sin_a = np.cos(deflection), np.sin(deflection)
                rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                surface_normal = rot_z @ initial_normal
            else:  # Left/Right - rotate around Y axis
                initial_normal = np.array([0, 0, 1]) if idx == 2 else np.array([0, 0, -1])
                cos_a, sin_a = np.cos(deflection), np.sin(deflection)
                rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
                surface_normal = rot_y @ initial_normal
            
            # Convert surface normal to world frame
            R = self.get_rotation_matrix()
            surface_normal_world = R @ surface_normal
            
            # Calculate angle of attack
            cos_aoa = np.dot(vel_dir, surface_normal_world)
            sin_aoa = np.sqrt(max(0, 1 - cos_aoa**2))
            aoa = np.arcsin(sin_aoa) * np.sign(cos_aoa)
            
            # Calculate lift coefficient with stall
            if abs(aoa) < self.canard_stall_angle:
                Cl = self.fin_lift_slope * aoa
            else:
                Cl = self.fin_lift_slope * self.canard_stall_angle * np.sign(aoa) * 0.5
            
            # Lift direction
            if sin_aoa > 0.01:
                lift_dir = surface_normal_world - vel_dir * cos_aoa
                lift_dir = lift_dir / np.linalg.norm(lift_dir)
            else:
                lift_dir = surface_normal_world
            
            lift_magnitude = q * Cl * self.fin_area
            lift_force = lift_dir * lift_magnitude
            
            # Drag on the surface
            surface_drag = -vel_dir * q * 0.1 * self.fin_area * (sin_aoa**2)
            
            total_surface_force = lift_force + surface_drag
            total_force += total_surface_force
            self.fin_forces.append(total_surface_force)
            
            # Calculate moment around CoG
            force_position_world = R @ surface_center
            moment_arm = force_position_world
            total_moment += np.cross(moment_arm, total_surface_force)
        
        # Angular damping (air resistance to rotation)
        damping_moment = -self.angular_velocity * 0.5 * rho * speed * 0.1
        total_moment += damping_moment
        
        return total_force / self.missile_mass, total_moment / self.moment_of_inertia
    
    def guidance_law(self):
        """Guidance law - returns desired canard deflection angles [top, bottom, left, right]"""
        # Vector from missile to target
        r = self.target_pos - self.missile_pos
        distance = np.linalg.norm(r)
        
        # Store for debug visualization
        self.debug_los_vector = r
        self.debug_distance = distance
        
        # Hit detection
        if distance < 50.0:
            print(f"HIT! Distance: {distance:.1f}m")
            self.missile_active = False
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Get missile orientation axes
        forward, right, up = self.get_body_axes()
        
        # Line of sight unit vector
        los = r / distance
        
        # Closing velocity (rate of decrease of distance)
        v_closing = -np.dot(self.missile_vel - self.target_vel, los)
        self.debug_v_closing = v_closing
        
        # Line of sight rate (angular velocity of LOS)
        # los_rate = (v_rel - los * v_closing) / distance
        v_rel = self.target_vel - self.missile_vel
        los_rate = (v_rel - los * np.dot(v_rel, los)) / distance
        
        # Proportional navigation: acceleration perpendicular to LOS
        # a_cmd = N * v_closing * los_rate
        speed = np.linalg.norm(self.missile_vel)
        if v_closing > 10.0 and speed > 10.0:
            a_cmd = self.guidance_gain * v_closing * los_rate
            self.debug_guidance_accel = a_cmd
        else:
            # If not closing or slow, just point at target
            a_cmd = (los - forward) * 10.0
            self.debug_guidance_accel = a_cmd
        
        # Convert acceleration command to body frame
        a_up = np.dot(a_cmd, up)  # Pitch control
        a_right = np.dot(a_cmd, right)  # Yaw control
        
        # Simple P controller: deflection proportional to desired acceleration
        # Positive a_up (want to go up) -> top canard positive, bottom negative
        # Positive a_right (want to go right) -> left canard positive, right negative
        
        gain = 0.002  # Deflection per m/s² (tune this)
        
        pitch_cmd = a_up * gain
        yaw_cmd = a_right * gain
        
        canard_cmds = np.array([
            pitch_cmd,   # top
            -pitch_cmd,  # bottom (opposite)
            yaw_cmd,     # left
            -yaw_cmd     # right (opposite)
        ])
        
        # Clamp to max deflection
        canard_cmds = np.clip(canard_cmds, -self.canard_max_deflection, self.canard_max_deflection)
        
        return canard_cmds
    
    def update(self):
        """Update simulation"""
        # Apply simulation speed multiplier
        effective_dt = self.dt * self.sim_speed
        
        # Update target (circular flight)
        self.target_angle += self.target_angular_speed * effective_dt
        self.target_pos[0] = self.target_radius * np.cos(self.target_angle)
        self.target_pos[1] = self.target_radius * np.sin(self.target_angle)
        self.target_pos[2] = self.target_altitude
        
        self.target_vel[0] = -self.target_radius * self.target_angular_speed * np.sin(self.target_angle)
        self.target_vel[1] = self.target_radius * self.target_angular_speed * np.cos(self.target_angle)
        self.target_vel[2] = 0.0
        
        # Only update path occasionally for performance
        if self.frame_count % self.path_update_interval == 0:
            self.target_path.append(self.target_pos.copy())
            if len(self.target_path) > 100:  # Reduced from 200
                self.target_path.pop(0)
        
        # Update missile
        if self.missile_active:
            self.missile_time += effective_dt
            
            # Calculate guidance command (returns 4 canard deflection angles)
            canard_cmd = self.guidance_law()
            
            # Smooth canard deflection (rate limit for each)
            max_deflection_rate = np.radians(180) * effective_dt  # 180 deg/s max rate
            for i in range(4):
                deflection_change = canard_cmd[i] - self.canard_deflection[i]
                deflection_change = np.clip(deflection_change, -max_deflection_rate, max_deflection_rate)
                self.canard_deflection[i] += deflection_change
            
            # Keep rear fins neutral for passive stability (or add active control here)
            self.fin_deflection = np.array([0.0, 0.0, 0.0, 0.0])
            
            # Get current speed
            speed = np.linalg.norm(self.missile_vel)
            
            # Calculate aerodynamic forces and moments
            aero_accel, angular_accel = self.calculate_aero_forces(speed)
            angular_accel = np.array(angular_accel)  # Ensure it's mutable
            
            # Apply thrust if fuel available (thrust along body axis)
            if self.missile_time < self.missile_fuel:
                forward, _, _ = self.get_body_axes()
                thrust_accel = forward * self.missile_thrust
                
                # Add small roll disturbance from thrust asymmetry/turbulence
                # This simulates real-world imperfections that cause roll
                if speed > 50.0:  # Only at higher speeds
                    roll_disturbance = np.random.normal(0, 0.005)  # Small random roll torque
                    angular_accel[2] += roll_disturbance
            else:
                thrust_accel = np.array([0.0, 0.0, 0.0])
            
            # Total acceleration
            total_accel = aero_accel + thrust_accel + self.gravity
            
            # Update angular velocity and orientation
            self.angular_velocity += angular_accel * effective_dt
            
            # Update Euler angles from angular velocity
            self.missile_pitch += self.angular_velocity[0] * effective_dt
            self.missile_yaw += self.angular_velocity[1] * effective_dt
            self.missile_roll += self.angular_velocity[2] * effective_dt
            
            # Keep pitch in reasonable range
            self.missile_pitch = np.clip(self.missile_pitch, -np.pi/2, np.pi/2)
            
            # Invalidate rotation matrix cache
            self._cached_orientation = None
            
            # Update velocity and position
            self.missile_vel += total_accel * effective_dt
            
            # Predict next position
            next_pos = self.missile_pos + self.missile_vel * effective_dt
            
            # Check ground collision BEFORE updating
            if next_pos[2] <= 0:
                # Hit ground - check if it's a slow landing or crash
                if self.missile_vel[2] < -10.0:  # Descending faster than 10 m/s
                    self.missile_pos[2] = 0
                    self.missile_active = False
                    print("MISSILE IMPACT GROUND")
                else:
                    # Prevent going underground - bounce or stop
                    self.missile_pos[0] = next_pos[0]
                    self.missile_pos[1] = next_pos[1]
                    self.missile_pos[2] = max(1.0, next_pos[2])  # Keep at least 1m above ground
                    self.missile_vel[2] = max(0, self.missile_vel[2])  # Stop downward velocity
            else:
                # Normal position update
                self.missile_pos = next_pos
            
            # Check bounds
            if (abs(self.missile_pos[0]) > self.bounds * 2 or 
                abs(self.missile_pos[1]) > self.bounds * 2 or 
                self.missile_pos[2] > self.bounds * 4):
                self.missile_active = False
                print("MISSILE OUT OF BOUNDS")
            
            # Only update path occasionally for performance
            if self.frame_count % self.path_update_interval == 0:
                self.missile_path.append(self.missile_pos.copy())
                if len(self.missile_path) > 100:  # Reduced from 200
                    self.missile_path.pop(0)
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update plot"""
        # Target
        self.target_dot.set_data([self.target_pos[0]/1000.0], [self.target_pos[1]/1000.0])
        self.target_dot.set_3d_properties([self.target_pos[2]/1000.0])
        
        if len(self.target_path) > 1:
            path_array = np.array(self.target_path) / 1000.0
            self.target_line.set_data(path_array[:, 0], path_array[:, 1])
            self.target_line.set_3d_properties(path_array[:, 2])
        
        # Missile
        if self.missile_spawned:
            self.missile_dot.set_data([self.missile_pos[0]/1000.0], [self.missile_pos[1]/1000.0])
            self.missile_dot.set_3d_properties([self.missile_pos[2]/1000.0])
            
            if len(self.missile_path) > 1:
                path_array = np.array(self.missile_path) / 1000.0
                self.missile_line.set_data(path_array[:, 0], path_array[:, 1])
                self.missile_line.set_3d_properties(path_array[:, 2])
        
        # Debug visualization (only when missile is active)
        if self.missile_active:
            # Line of sight from missile to target
            los_start = self.missile_pos / 1000.0
            los_end = self.target_pos / 1000.0
            self.los_line.set_data([los_start[0], los_end[0]], [los_start[1], los_end[1]])
            self.los_line.set_3d_properties([los_start[2], los_end[2]])
            
            # Missile velocity vector (scaled for visibility)
            vel_scale = 0.5  # Scale factor for velocity vector display
            missile_vel_end = self.missile_pos + self.missile_vel * vel_scale
            self.missile_vel_line.set_data([self.missile_pos[0]/1000.0, missile_vel_end[0]/1000.0],
                                           [self.missile_pos[1]/1000.0, missile_vel_end[1]/1000.0])
            self.missile_vel_line.set_3d_properties([self.missile_pos[2]/1000.0, missile_vel_end[2]/1000.0])
            
            # Target velocity vector
            target_vel_end = self.target_pos + self.target_vel * vel_scale
            self.target_vel_line.set_data([self.target_pos[0]/1000.0, target_vel_end[0]/1000.0],
                                          [self.target_pos[1]/1000.0, target_vel_end[1]/1000.0])
            self.target_vel_line.set_3d_properties([self.target_pos[2]/1000.0, target_vel_end[2]/1000.0])
            
            # Guidance acceleration vector
            if hasattr(self, 'debug_guidance_accel'):
                guide_scale = 50.0  # Scale for visibility (increased from 5.0)
                guide_end = self.missile_pos + self.debug_guidance_accel * guide_scale
                self.guidance_vec_line.set_data([self.missile_pos[0]/1000.0, guide_end[0]/1000.0],
                                                [self.missile_pos[1]/1000.0, guide_end[1]/1000.0])
                self.guidance_vec_line.set_3d_properties([self.missile_pos[2]/1000.0, guide_end[2]/1000.0])
        else:
            # Clear debug lines when missile not active
            self.los_line.set_data([], [])
            self.los_line.set_3d_properties([])
            self.missile_vel_line.set_data([], [])
            self.missile_vel_line.set_3d_properties([])
            self.target_vel_line.set_data([], [])
            self.target_vel_line.set_3d_properties([])
            self.guidance_vec_line.set_data([], [])
            self.guidance_vec_line.set_3d_properties([])
        
        # Info text
        if self.missile_active:
            distance = np.linalg.norm(self.target_pos - self.missile_pos) / 1000.0
            speed_kmh = np.linalg.norm(self.missile_vel) * 3.6
            fuel_left = max(0, self.missile_fuel - self.missile_time)
            alt_km = self.missile_pos[2] / 1000.0
            info = f"ACTIVE | Dist: {distance:.2f}km | Speed: {speed_kmh:.0f}km/h\n"
            info += f"Alt: {alt_km:.2f}km | Fuel: {fuel_left:.1f}s | Time: {self.missile_time:.1f}s\n"
            if hasattr(self, 'debug_v_closing'):
                info += f"Closing: {self.debug_v_closing:.1f}m/s | "
            info += f"SimSpeed: {self.sim_speed:.1f}x\n"
            info += f"Canards: T{np.degrees(self.canard_deflection[0]):+.0f}° B{np.degrees(self.canard_deflection[1]):+.0f}° L{np.degrees(self.canard_deflection[2]):+.0f}° R{np.degrees(self.canard_deflection[3]):+.0f}°"
        elif self.missile_spawned:
            distance = np.linalg.norm(self.target_pos - self.missile_pos) / 1000.0
            info = f"READY | Distance: {distance:.2f}km\n"
            info += f"SPACE=Launch | R=Reset | Scroll=Speed | SimSpeed: {self.sim_speed:.1f}x"
        else:
            info = f"SPACE=Spawn | Scroll=Speed | SimSpeed: {self.sim_speed:.1f}x"
        
        self.info_text.set_text(info)
        
        # Update display (closeup disabled for performance)
        self.frame_count += 1
        
        # Only redraw when needed
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def rotate_mesh(self, faces, angle, axis, center):
        """Rotate mesh faces around an axis through the object's center"""
        if axis == 'y':  # Rotation around Y-axis (for top/bottom surfaces)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        elif axis == 'z':  # Rotation around Z-axis (for left/right surfaces)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        else:
            return faces
        
        rotated_faces = []
        for face in faces:
            rotated_face = []
            for vertex in face:
                # Translate to origin (center of object)
                v = np.array(vertex) - center
                # Rotate
                v_rot = rot @ v
                # Translate back
                v_final = v_rot + center
                rotated_face.append(v_final)
            rotated_faces.append(rotated_face)
        return rotated_faces
    
    def get_rotation_matrix(self, use_cache=True):
        """Get rotation matrix for missile orientation (pitch, yaw, roll)"""
        # Check cache
        current_orientation = (self.missile_pitch, self.missile_yaw, self.missile_roll)
        if use_cache and self._cached_orientation == current_orientation:
            return self._cached_rotation_matrix
        
        # Rotation around Z-axis (yaw)
        cos_yaw, sin_yaw = np.cos(self.missile_yaw), np.sin(self.missile_yaw)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Rotation around Y-axis (pitch)
        cos_pitch, sin_pitch = np.cos(self.missile_pitch), np.sin(self.missile_pitch)
        R_pitch = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        # Rotation around X-axis (roll)
        cos_roll, sin_roll = np.cos(self.missile_roll), np.sin(self.missile_roll)
        R_roll = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        # Combined rotation: first roll, then pitch, then yaw
        R = R_yaw @ R_pitch @ R_roll
        
        # Cache result
        if use_cache:
            self._cached_rotation_matrix = R
            self._cached_orientation = current_orientation
        
        return R
    
    def transform_mesh_to_world(self, faces):
        """Transform mesh from body frame to world frame showing missile orientation"""
        R = self.get_rotation_matrix()
        transformed_faces = []
        for face in faces:
            transformed_face = []
            for vertex in face:
                v = np.array(vertex)
                v_transformed = R @ v
                transformed_face.append(v_transformed)
            transformed_faces.append(transformed_face)
        return transformed_faces
    
    def update_closeup(self):
        """Closeup view disabled for performance"""
        return  # Disabled for performance
        
        if not self.missile_spawned:
            # Clear closeup when not spawned
            self.closeup_body.set_segments([])
            self.closeup_canard_top.set_verts([])
            self.closeup_canard_bottom.set_verts([])
            self.closeup_canard_left.set_verts([])
            self.closeup_canard_right.set_verts([])
            self.closeup_fin_top.set_verts([])
            self.closeup_fin_bottom.set_verts([])
            self.closeup_fin_left.set_verts([])
            self.closeup_fin_right.set_verts([])
            self.closeup_velocity_arrow.set_data([], [])
            self.closeup_velocity_arrow.set_3d_properties([])
            # Clear aero force vectors
            for line in [self.closeup_aero_canard_top, self.closeup_aero_canard_bottom,
                        self.closeup_aero_canard_left, self.closeup_aero_canard_right,
                        self.closeup_aero_fin_top, self.closeup_aero_fin_bottom,
                        self.closeup_aero_fin_left, self.closeup_aero_fin_right]:
                line.set_data([], [])
                line.set_3d_properties([])
            self.closeup_text.set_text('')
            return
        
        # Get body axes
        forward, right, up = self.get_body_axes()
        
        # DISABLED: Body wireframe is too expensive for real-time rendering
        # Just clear it to avoid confusion
        if self.frame_count == 1:
            self.closeup_body.set_segments([])
        
        # Canards from OBJ with rotation - simplified rendering
        # Only update if significant change in deflection
        if self.frame_count % (self.closeup_update_interval * 2) == 0:
            canard_objects = [
                ('Can_TOP', self.closeup_canard_top, self.canard_deflection[0], 'z'),
                ('Can_DOWN', self.closeup_canard_bottom, self.canard_deflection[1], 'z'),
                ('Can_LEFT', self.closeup_canard_left, self.canard_deflection[2], 'y'),
                ('Can_RIGHT', self.closeup_canard_right, self.canard_deflection[3], 'y')
            ]
            
            for obj_name, poly, angle, axis in canard_objects:
                if obj_name in self.obj_data and obj_name in self.obj_centers:
                    # First apply canard deflection
                    rotated_faces = self.rotate_mesh(self.obj_data[obj_name], angle, axis, self.obj_centers[obj_name])
                    # Then apply missile orientation
                    transformed_faces = self.transform_mesh_to_world(rotated_faces)
                    poly.set_verts(transformed_faces)
        
        # Rear fins from OBJ with rotation - simplified rendering
        if self.frame_count % (self.closeup_update_interval * 2) == 0:
            fin_objects = [
                ('Stab_TOP', self.closeup_fin_top, self.fin_deflection[0], 'z'),
                ('Stab_DOWN', self.closeup_fin_bottom, self.fin_deflection[1], 'z'),
                ('Stab_LEFT', self.closeup_fin_left, self.fin_deflection[2], 'y'),
                ('Stab_RIGHT', self.closeup_fin_right, self.fin_deflection[3], 'y')
            ]
            
            for obj_name, poly, angle, axis in fin_objects:
                if obj_name in self.obj_data and obj_name in self.obj_centers:
                    # First apply fin deflection
                    rotated_faces = self.rotate_mesh(self.obj_data[obj_name], angle, axis, self.obj_centers[obj_name])
                    # Then apply missile orientation
                    transformed_faces = self.transform_mesh_to_world(rotated_faces)
                    poly.set_verts(transformed_faces)
        
        # Velocity arrow (relative to body frame)
        speed = np.linalg.norm(self.missile_vel)
        if speed > 1.0:
            vel_dir = self.missile_vel / speed
            # Transform to body frame
            vel_body = np.array([
                np.dot(vel_dir, forward),
                np.dot(vel_dir, right),
                np.dot(vel_dir, up)
            ])
            vel_scale = 0.5
            vel_arrow_end = vel_body * vel_scale
            self.closeup_velocity_arrow.set_data([0, vel_arrow_end[0]], [0, vel_arrow_end[1]])
            self.closeup_velocity_arrow.set_3d_properties([0, vel_arrow_end[2]])
        
        # Draw aero force vectors - SIMPLIFIED for performance
        # Only update every few closeup frames
        if self.frame_count % (self.closeup_update_interval * 3) != 0:
            return  # Skip expensive force vector updates
        
        force_scale = 0.002  # Scale forces for visibility
        canard_aero_lines = [self.closeup_aero_canard_top, self.closeup_aero_canard_bottom,
                            self.closeup_aero_canard_left, self.closeup_aero_canard_right]
        fin_aero_lines = [self.closeup_aero_fin_top, self.closeup_aero_fin_bottom,
                         self.closeup_aero_fin_left, self.closeup_aero_fin_right]
        
        # Get rotation matrix for transforming force vectors
        R = self.get_rotation_matrix()
        
        # Canard force vectors (from center of canard surface, along actual surface normal)
        if hasattr(self, 'canard_forces') and len(self.canard_forces) == 4 and hasattr(self, 'obj_centers'):
            canard_axes = ['z', 'z', 'y', 'y']  # rotation axes
            
            for i, (line, force) in enumerate(zip(canard_aero_lines, self.canard_forces)):
                obj_name = ['Can_TOP', 'Can_DOWN', 'Can_LEFT', 'Can_RIGHT'][i]
                if obj_name in self.obj_centers and np.linalg.norm(force) > 0.1:
                    # Use center of canard surface as application point
                    center = np.array(self.obj_centers[obj_name])
                    angle = self.canard_deflection[i]
                    
                    # Calculate actual surface normal based on deflection
                    # Top/Bottom canards rotate around Z, so normal rotates in XY plane
                    # Left/Right canards rotate around Y, so normal rotates in XZ plane
                    if i < 2:  # Top/Bottom - rotate around Z
                        # Initial normal for top/bottom is along Y axis
                        initial_normal = np.array([0, 1, 0]) if i == 0 else np.array([0, -1, 0])
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                        normal = rot_z @ initial_normal
                    else:  # Left/Right - rotate around Y
                        # Initial normal for left/right is along Z axis
                        initial_normal = np.array([0, 0, 1]) if i == 2 else np.array([0, 0, -1])
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
                        normal = rot_y @ initial_normal
                    
                    # Project force onto surface normal
                    force_along_normal = np.dot(force, normal) * normal
                    
                    # Apply missile rotation to both center and end points
                    center_transformed = R @ center
                    end = center + force_along_normal * force_scale
                    end_transformed = R @ end
                    
                    line.set_data([center_transformed[0], end_transformed[0]], [center_transformed[1], end_transformed[1]])
                    line.set_3d_properties([center_transformed[2], end_transformed[2]])
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])
        
        # Fin force vectors (from center of fin surface, along actual surface normal)
        if hasattr(self, 'fin_forces') and len(self.fin_forces) == 4 and hasattr(self, 'obj_centers'):
            for i, (line, force) in enumerate(zip(fin_aero_lines, self.fin_forces)):
                obj_name = ['Stab_TOP', 'Stab_DOWN', 'Stab_LEFT', 'Stab_RIGHT'][i]
                if obj_name in self.obj_centers and np.linalg.norm(force) > 0.1:
                    # Use center of fin surface as application point
                    center = np.array(self.obj_centers[obj_name])
                    angle = self.fin_deflection[i]
                    
                    # Calculate actual surface normal based on deflection
                    if i < 2:  # Top/Bottom - rotate around Z
                        # Initial normal for top/bottom is along Y axis
                        initial_normal = np.array([0, 1, 0]) if i == 0 else np.array([0, -1, 0])
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                        normal = rot_z @ initial_normal
                    else:  # Left/Right - rotate around Y
                        # Initial normal for left/right is along Z axis
                        initial_normal = np.array([0, 0, 1]) if i == 2 else np.array([0, 0, -1])
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
                        normal = rot_y @ initial_normal
                    
                    # Project force onto surface normal
                    force_along_normal = np.dot(force, normal) * normal
                    
                    # Apply missile rotation to both center and end points
                    center_transformed = R @ center
                    end = center + force_along_normal * force_scale
                    end_transformed = R @ end
                    
                    line.set_data([center_transformed[0], end_transformed[0]], [center_transformed[1], end_transformed[1]])
                    line.set_3d_properties([center_transformed[2], end_transformed[2]])
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])
        
        # Closeup info text
        info = "ORIENTATION:\n"
        info += f"Pitch: {np.degrees(self.missile_pitch):+.1f}°\n"
        info += f"Yaw:   {np.degrees(self.missile_yaw):+.1f}°\n"
        info += f"Roll:  {np.degrees(self.missile_roll):+.1f}°\n\n"
        info += "CANARDS (front)\n"
        info += f"Top:    {np.degrees(self.canard_deflection[0]):+.1f}°"
        if abs(self.canard_deflection[0]) > self.canard_stall_angle:
            info += " STALL"
        info += "\n"
        info += f"Bottom: {np.degrees(self.canard_deflection[1]):+.1f}°"
        if abs(self.canard_deflection[1]) > self.canard_stall_angle:
            info += " STALL"
        info += "\n"
        info += f"Left:   {np.degrees(self.canard_deflection[2]):+.1f}°"
        if abs(self.canard_deflection[2]) > self.canard_stall_angle:
            info += " STALL"
        info += "\n"
        info += f"Right:  {np.degrees(self.canard_deflection[3]):+.1f}°"
        if abs(self.canard_deflection[3]) > self.canard_stall_angle:
            info += " STALL"
        info += "\n\nFINS (rear)\n"
        info += f"T:{np.degrees(self.fin_deflection[0]):+.0f}° "
        info += f"B:{np.degrees(self.fin_deflection[1]):+.0f}° "
        info += f"L:{np.degrees(self.fin_deflection[2]):+.0f}° "
        info += f"R:{np.degrees(self.fin_deflection[3]):+.0f}°\n"
        info += f"\nAngular vel:\n"
        info += f"  Pitch: {np.degrees(self.angular_velocity[0]):+.1f}°/s\n"
        info += f"  Yaw:   {np.degrees(self.angular_velocity[1]):+.1f}°/s\n"
        info += f"  Roll:  {np.degrees(self.angular_velocity[2]):+.1f}°/s"
        
        self.closeup_text.set_text(info)
    
    def reset(self):
        """Reset simulation"""
        self.missile_active = False
        self.missile_spawned = True
        self.missile_pos = np.array([0.0, 0.0, 50.0])
        self.missile_vel = np.array([0.0, 0.0, 0.0])
        self.missile_time = 0.0
        
        # Reset orientation to point straight up
        self.missile_pitch = -np.pi / 2.0  # 90 degrees up
        self.missile_yaw = 0.0
        self.missile_roll = 0.0
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
        self.missile_path = [self.missile_pos.copy()]
        print("SIMULATION RESET")
    
    def on_scroll(self, event):
        """Mouse scroll to change simulation speed"""
        if event.button == 'up':
            self.sim_speed = min(10.0, self.sim_speed + 0.25)
        elif event.button == 'down':
            self.sim_speed = max(0.1, self.sim_speed - 0.25)
        print(f"Simulation speed: {self.sim_speed:.2f}x")
    
    def on_key(self, event):
        """Keyboard controls"""
        if event.key == ' ':
            if not self.missile_spawned:
                # Spawn missile
                self.missile_spawned = True
                self.missile_pos = np.array([0.0, 0.0, 50.0])
                self.missile_vel = np.array([0.0, 0.0, 0.0])
                self.missile_pitch = -np.pi / 2.0  # Point straight up
                self.missile_yaw = 0.0
                self.missile_roll = 0.0
                self.missile_path = [self.missile_pos.copy()]
                print("MISSILE SPAWNED - Press SPACE again to launch")
            else:
                # Launch missile
                self.launch_missile()
        elif event.key == 'r':
            self.reset()
        elif event.key == '+':
            self.sim_speed = min(10.0, self.sim_speed + 0.5)
            print(f"Simulation speed: {self.sim_speed:.2f}x")
        elif event.key == '-':
            self.sim_speed = max(0.1, self.sim_speed - 0.5)
            print(f"Simulation speed: {self.sim_speed:.2f}x")
    
    def show(self):
        print("=" * 50)
        print("MISSILE GUIDANCE SIMULATION")
        print("=" * 50)
        print("\nScenario:")
        print("  Target: Aircraft flying at 700 km/h in 5km radius circle at 5km altitude")
        print("  Missile: Launches from (0,0,0) with proportional navigation")
        print("\nMissile Specs:")
        print(f"  Mass: {self.missile_mass:.0f} kg")
        print(f"  Thrust: {self.missile_thrust:.0f} m/s²")
        print(f"  Fuel: {self.missile_fuel:.0f} seconds")
        print(f"  Body Drag Coeff: {self.body_drag_coeff}")
        print(f"  Canard Area: {self.canard_area} m² (max deflection: {np.degrees(self.canard_max_deflection):.0f}°)")
        print(f"  Fin Area: {self.fin_area} m²")
        print(f"  Rolleron Area: {self.rolleron_area} m² (roll damping)")
        print(f"  Canard Stall Angle: {np.degrees(self.canard_stall_angle):.0f}°")
        print(f"  Guidance: Proportional Navigation (N={self.guidance_gain})")
        print("\nControls:")
        print("  SPACE      - Launch missile")
        print("  R          - Reset simulation")
        print("  Scroll/+/- - Change sim speed (0.1x to 10x)")
        print("\nPhysics:")
        print(f"  Time step: {self.dt} s")
        print(f"  Gravity: {abs(self.gravity[2]):.2f} m/s²")
        print(f"  Roll Damping: Active (Rollerons)")
        print("=" * 50)
        plt.show()

if __name__ == "__main__":
    sim = MissileSimulation()
    sim.show()

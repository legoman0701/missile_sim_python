import numpy as np
import pygame
import sys
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class MissileSimulation:
    def __init__(self):
        # Initialize Pygame and OpenGL
        pygame.init()
        self.width, self.height = 1600, 900
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption('Missile Simulation - PyGame/OpenGL (60 FPS)')
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Camera setup
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 50000.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Camera position and rotation
        self.camera_distance = 15000.0
        self.camera_angle_h = 0.0
        self.camera_angle_v = 30.0
        self.follow_missile = True
        
        # Physics parameters
        self.bounds = 10000.0
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.dt = 1.0 / 60.0  # 60 FPS
        
        # Target (aircraft)
        self.target_pos = np.array([5000.0, 0.0, 5000.0])
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.target_altitude = 5000.0
        self.target_radius = 5000.0
        target_speed = 700.0 * 1000.0 / 3600.0
        self.target_angular_speed = target_speed / self.target_radius
        self.target_angle = 0.0
        
        # Missile parameters
        self.missile_pos = np.array([0.0, 0.0, 50.0])
        self.missile_vel = np.array([0.0, 0.0, 0.0])
        self.missile_spawned = True
        self.missile_active = False
        self.missile_thrust = 150.0
        self.missile_mass = 85.0
        self.missile_fuel = 20.0
        self.missile_time = 0.0
        
        # Aerodynamic parameters
        self.body_drag_coeff = 0.3
        self.body_area = 0.02
        self.canard_area = 0.03
        self.canard_deflection = np.array([0.0, 0.0, 0.0, 0.0])
        self.canard_max_deflection = np.radians(25)
        self.canard_distance = 0.5
        self.canard_lift_slope = 5.0
        self.canard_stall_angle = np.radians(15)
        
        self.fin_area = 0.04
        self.fin_deflection = np.array([0.0, 0.0, 0.0, 0.0])
        self.fin_distance = -1.2  # BEHIND CoG for stability (negative X in body frame)
        self.fin_lift_slope = 3.5
        
        # Missile orientation using quaternions (avoids gimbal lock)
        # Initialize pointing up (pitch = -90 degrees)
        self.missile_quat = self.euler_to_quaternion(0.0, -np.pi / 2.0, 0.0)
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # [pitch_rate, yaw_rate, roll_rate]
        self.moment_of_inertia_long = 0.3 # kg*m^2
        self.moment_of_inertia_trns = 65.0 # kg*m^2
        
        self.canard_forces = [0.0, 0.0, 0.0, 0.0]
        self.fin_forces = [0.0, 0.0, 0.0, 0.0]
        
        # Guidance
        self.guidance_gain = 3
        
        # Load OBJ model
        self.obj_data = self.load_obj_model('missle.obj')
        self.obj_centers = self.calculate_obj_centers()
        
        # Simulation speed
        self.sim_speed = 1.0
        
        # Path history
        self.target_path = [self.target_pos.copy()]
        self.missile_path = [self.missile_pos.copy()]
        self.max_path_length = 300
        
        # Font for HUD
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.fps = 60
        
        # Debug output
        self.debug_enabled = False
        self.debug_counter = 0
        
        # Auto-launch and auto-close
        self.auto_close_time = 999.0  # Disabled by default
        self.auto_launch_delay = 999.0  # Disabled by default  
        self.missile_launched = False
        self.start_time = None
        
        # Mouse control
        self.mouse_down = False
        self.last_mouse_pos = None
        
        print("=" * 60)
        print("MISSILE SIMULATION - PYGAME/OPENGL")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE      - Launch missile")
        print("  R          - Reset simulation")
        print("  F          - Toggle follow missile camera")
        print("  D          - Toggle debug output")
        print("  +/-        - Change sim speed")
        print("  Mouse Drag - Rotate camera")
        print("  Scroll     - Zoom in/out")
        print("\nTarget: 700 km/h circular flight at 5km altitude")
        print("Missile: Proportional navigation guidance")
        print("=" * 60)
    
    def load_obj_model(self, filepath):
        """Load OBJ file and parse mesh data by object name"""
        import os
        obj_path = os.path.join(os.path.dirname(__file__), filepath)
        
        vertices = []
        objects = {}
        current_obj = None
        
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):  # Vertex
                        parts = line.split()
                        # Blender Z-up to our coordinate system
                        # Swap axes: Blender X,Y,Z -> our Z,X,Y (missile length along X)
                        vertices.append([float(parts[3]), float(parts[1]), float(parts[2])])
                    elif line.startswith('o '):  # Object name
                        current_obj = line.split()[1]
                        objects[current_obj] = []
                    elif line.startswith('f ') and current_obj:  # Face
                        parts = line.split()[1:]
                        face_verts = []
                        for part in parts:
                            idx = int(part.split('/')[0]) - 1  # OBJ indices are 1-based
                            face_verts.append(vertices[idx])
                        objects[current_obj].append(face_verts)
        except FileNotFoundError:
            print(f"Warning: OBJ file '{filepath}' not found. Using simple missile model.")
            return {}
        
        print(f"Loaded OBJ with {len(objects)} objects: {list(objects.keys())}")
        for obj_name, faces in objects.items():
            print(f"  {obj_name}: {len(faces)} faces")
        
        return objects
    
    def calculate_obj_centers(self):
        """Calculate center of each object for rotation pivots"""
        obj_centers = {}
        for obj_name, faces in self.obj_data.items():
            all_verts = []
            for face in faces:
                all_verts.extend(face)
            if all_verts:
                center = np.mean(all_verts, axis=0)
                obj_centers[obj_name] = center
        return obj_centers
    
    def get_surface_normal(self, obj_name, deflection_angle=0.0, rotation_axis=None):
        """Calculate surface normal for a control surface after deflection"""
        if obj_name not in self.obj_data or len(self.obj_data[obj_name]) == 0:
            return np.array([0.0, 1.0, 0.0])  # Default normal
        
        # Get first face to calculate normal
        face = self.obj_data[obj_name][0]
        if len(face) < 3:
            return np.array([0.0, 1.0, 0.0])
        
        # Calculate face normal using cross product
        v0 = np.array(face[0])
        v1 = np.array(face[1])
        v2 = np.array(face[2])
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize
        norm_length = np.linalg.norm(normal)
        if norm_length > 0.001:
            normal = normal / norm_length
        else:
            normal = np.array([0.0, 1.0, 0.0])
        
        # Apply deflection rotation to the normal
        if deflection_angle != 0.0 and rotation_axis is not None:
            if rotation_axis == 'y':  # Left/Right canards rotate around Y
                cos_a, sin_a = np.cos(deflection_angle), np.sin(deflection_angle)
                rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
                normal = rot_y @ normal
            elif rotation_axis == 'z':  # Top/Bottom canards rotate around Z
                cos_a, sin_a = np.cos(deflection_angle), np.sin(deflection_angle)
                rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                normal = rot_z @ normal
        
        return normal
    
    def euler_to_quaternion(self, yaw, pitch, roll):
        """Convert Euler angles to quaternion [w, x, y, z]"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def quaternion_to_euler(self, q):
        """Convert quaternion [w, x, y, z] to Euler angles (yaw, pitch, roll)"""
        w, x, y, z = q
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        return yaw, pitch, roll
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])
    
    def normalize_quaternion(self, q):
        """Normalize a quaternion"""
        norm = np.linalg.norm(q)
        if norm > 0.0001:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def get_rotation_matrix(self):
        """Get rotation matrix from quaternion"""
        w, x, y, z = self.missile_quat
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def get_body_axes(self):
        """Get missile body axes from quaternion"""
        R = self.get_rotation_matrix()
        
        # Body axes are columns of rotation matrix
        forward = R[:, 0]  # X-axis (forward)
        right = R[:, 1]    # Y-axis (right)
        up = R[:, 2]       # Z-axis (up)
        
        return forward, right, up
    
    def guidance_law(self):
        # """Proportional navigation guidance"""
        # r = self.target_pos - self.missile_pos
        # distance = np.linalg.norm(r)
        # 
        # if distance < 50.0:
        #     print(f"HIT! Distance: {distance:.1f}m")
        #     self.missile_active = False
        #     return np.array([0.0, 0.0, 0.0, 0.0])
        # 
        # forward, right, up = self.get_body_axes()
        # los = r / distance
        # 
        # v_closing = -np.dot(self.missile_vel - self.target_vel, los)
        # v_rel = self.target_vel - self.missile_vel
        # los_rate = (v_rel - los * np.dot(v_rel, los)) / distance
        # 
        # speed = np.linalg.norm(self.missile_vel)
        # if v_closing > 10.0 and speed > 10.0:
        #     a_cmd = self.guidance_gain * v_closing * los_rate
        # else:
        #     a_cmd = (los - forward) * 10.0
        # 
        # a_up = np.dot(a_cmd, up)
        # a_right = np.dot(a_cmd, right)
        # 
        # gain = 0.002
        # pitch_cmd = a_up * gain
        # yaw_cmd = a_right * gain
        
        canard_cmds = np.array([
            0.1,
            0,
            0,
            0
        ])
        
        return np.clip(canard_cmds, -self.canard_max_deflection, self.canard_max_deflection)
    
    def calculate_lift(self, aoa, speed):
        """Calculate lift and drag forces"""
        rho = 1.225  # Air density at sea level (kg/m^3)
        q = 0.5 * rho * speed * speed  # Dynamic pressure
        lift = -aoa * q*0.002
        drag = aoa * q*0.002
        
        return lift, #drag
    
    def calculate_aero_forces(self, speed):
        # Apply missile orientation
        if 'Body' in self.obj_centers:
            body_center = self.obj_centers['Body']
            glTranslatef(body_center[0], body_center[1], body_center[2])
        
        yaw, pitch, roll = self.quaternion_to_euler(self.missile_quat)
        glRotatef(np.degrees(yaw), 0, 0, 1)
        glRotatef(np.degrees(pitch), 0, 1, 0)
        glRotatef(np.degrees(roll), 1, 0, 0)
        
        if 'Body' in self.obj_centers:
            body_center = self.obj_centers['Body']
            glTranslatef(-body_center[0], -body_center[1], -body_center[2])
        
        # Draw canard forces (magenta) - along surface normal
        canard_names = ['Can_TOP', 'Can_DOWN', 'Can_LEFT', 'Can_RIGHT']
        canard_axes = ['z', 'z', 'y', 'y']
        
        force_vectors = np.array([0.0, 0.0, 0.0])
        moments_world = np.array([0.0, 0.0, 0.0])
        # rotation matrix from body -> world
        R = self.get_rotation_matrix()
        for i, (obj_name, axis) in enumerate(zip(canard_names, canard_axes)):
            if obj_name in self.obj_centers and 'Body' in self.obj_centers:
                # Get vertex center (body coords)
                center = self.obj_centers[obj_name]

                # Get surface normal with deflection applied (body coords)
                normal = self.get_surface_normal(obj_name, self.canard_deflection[i], axis)

                # Compute fin position and velocity including rotational contribution (world frame)
                model_scale = 100.0
                r_body = (center - self.obj_centers['Body']) * model_scale
                omega_world = R @ self.angular_velocity
                r_world = R @ r_body
                relative_fin_vel = self.missile_vel + np.cross(omega_world, r_world)

                # Incoming flow direction (air relative to surface) is opposite of fin velocity
                speed_fin = np.linalg.norm(relative_fin_vel) + 1e-9
                flow_dir = -relative_fin_vel / speed_fin

                # Surface normal: convert to world and normalize
                normal_world = R @ normal
                normal_world = normal_world / (np.linalg.norm(normal_world) + 1e-9)

                # Angle between surface normal and incoming flow
                cos_ang = np.clip(np.dot(normal_world, flow_dir), -1.0, 1.0)
                aoa = np.arccos(cos_ang) - np.pi / 2.0
                if self.debug_enabled:
                    print(f"Canard {obj_name}: aoa={aoa:.6f}, cos={cos_ang:.6f}")

                # calculate lift (unpack first return element)
                lift, = self.calculate_lift(aoa, speed)
                force_magnitude = lift

                # Force vector in world coords
                force_vector_world = normal_world * force_magnitude

                # Accumulate forces and moments (moments in world frame)
                force_vectors += force_vector_world
                moments_world += np.cross(r_world, force_vector_world)

                self.canard_forces[i] = force_magnitude
             
             
        # Draw fin forces (cyan) - along surface normal
        fin_names = ['Stab_TOP', 'Stab_DOWN', 'Stab_LEFT', 'Stab_RIGHT']
        fin_axes = ['z', 'z', 'y', 'y']
        
        for i, (obj_name, axis) in enumerate(zip(fin_names, fin_axes)):
            if obj_name in self.obj_centers and 'Body' in self.obj_centers:
                # Get vertex center (body coords)
                center = self.obj_centers[obj_name]

                # Get surface normal with deflection applied (body coords)
                normal = self.get_surface_normal(obj_name, self.fin_deflection[i], axis)

                # Compute fin position and velocity including rotational contribution (world frame)
                model_scale = 100.0
                r_body = (center - self.obj_centers['Body']) * model_scale
                omega_world = R @ self.angular_velocity
                r_world = R @ r_body
                relative_fin_vel = self.missile_vel + np.cross(omega_world, r_world)

                # Incoming flow direction (air relative to surface) is opposite of fin velocity
                speed_fin = np.linalg.norm(relative_fin_vel) + 1e-9
                flow_dir = -relative_fin_vel / speed_fin

                # Surface normal: convert to world and normalize
                normal_world = R @ normal
                normal_world = normal_world / (np.linalg.norm(normal_world) + 1e-9)

                # Angle between surface normal and incoming flow
                cos_ang = np.clip(np.dot(normal_world, flow_dir), -1.0, 1.0)
                aoa = np.arccos(cos_ang) - np.pi / 2.0
                if self.debug_enabled:
                    print(f"Canard {obj_name}: aoa={aoa:.6f}, cos={cos_ang:.6f}")

                # calculate lift (unpack first return element)
                lift, = self.calculate_lift(aoa, speed)
                force_magnitude = lift

                # Force vector in world coords
                force_vector_world = normal_world * force_magnitude

                # Accumulate forces and moments (moments in world frame)
                force_vectors += force_vector_world
                moments_world += np.cross(r_world, force_vector_world)

                self.fin_forces[i] = force_magnitude
                    

        # Convert torque (moments) to angular acceleration using principal moments of inertia
        # Moments currently in world coords; convert to body coords before dividing by inertia
        inertia = np.array([
            self.moment_of_inertia_long,
            self.moment_of_inertia_trns,
            self.moment_of_inertia_trns
        ], dtype=float)

        # Avoid division by zero
        inertia = np.where(np.abs(inertia) < 1e-12, 1e-12, inertia)

        # Convert moments to body axes and compute angular acceleration (rad/s^2)
        moments_body = R.T @ moments_world
        angular_accel = moments_body / inertia

        # Convert accumulated forces (world frame) to acceleration (a = F/m)
        aero_accel = force_vectors / max(self.missile_mass, 1e-9)

        return np.array(aero_accel), np.array(angular_accel)
    
    def update_physics(self):
        """Update simulation physics"""
        effective_dt = self.dt * self.sim_speed
        
        # Update target
        self.target_angle += self.target_angular_speed * effective_dt
        self.target_pos[0] = self.target_radius * np.cos(self.target_angle)
        self.target_pos[1] = self.target_radius * np.sin(self.target_angle)
        self.target_pos[2] = self.target_altitude
        
        self.target_vel[0] = -self.target_radius * self.target_angular_speed * np.sin(self.target_angle)
        self.target_vel[1] = self.target_radius * self.target_angular_speed * np.cos(self.target_angle)
        
        if self.frame_count % 2 == 0:
            self.target_path.append(self.target_pos.copy())
            if len(self.target_path) > self.max_path_length:
                self.target_path.pop(0)
        
        # Update missile
        if self.missile_active:
            self.missile_time += effective_dt
            
            # Guidance
            canard_cmd = self.guidance_law()
            max_rate = np.radians(180) * effective_dt
            for i in range(4):
                change = canard_cmd[i] - self.canard_deflection[i]
                change = np.clip(change, -max_rate, max_rate)
                self.canard_deflection[i] += change
            
            self.fin_deflection = np.array([0.0, 0.0, 0.0, 0.0])
            
            speed = np.linalg.norm(self.missile_vel)
            aero_accel, angular_accel = self.calculate_aero_forces(speed)
            angular_accel = np.array(angular_accel)
            
            # Thrust
            if self.missile_time < self.missile_fuel:
                forward, _, _ = self.get_body_axes()
                # Convert thrust (N) to acceleration (m/s^2) using mass
                thrust_accel = forward * (self.missile_thrust / max(self.missile_mass, 1e-9))
                if speed > 50.0:
                    angular_accel[2] += np.random.normal(0, 0.005)
            else:
                thrust_accel = np.array([0.0, 0.0, 0.0])
            
            total_accel = aero_accel + thrust_accel + self.gravity
            
            # Update orientation using quaternions (no gimbal lock!)
            self.angular_velocity += angular_accel * effective_dt
            self.angular_velocity[2] -= self.angular_velocity[2] * effective_dt * 0.1  # Roll damping
            
            # Convert angular velocity to quaternion derivative
            # dq/dt = 0.5 * q * omega_quat
            omega_quat = np.array([0.0, self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2]])
            q_dot = 0.5 * self.quaternion_multiply(self.missile_quat, omega_quat)
            
            # Update quaternion
            self.missile_quat += q_dot * effective_dt
            
            # Normalize to prevent drift
            self.missile_quat = self.normalize_quaternion(self.missile_quat)
            
            # Update position
            self.missile_vel += total_accel * effective_dt
            next_pos = self.missile_pos + self.missile_vel * effective_dt
            
            if next_pos[2] <= 0:
                if self.missile_vel[2] < -10.0:
                    self.missile_pos[2] = 0
                    self.missile_active = False
                    print("MISSILE IMPACT GROUND")
                else:
                    self.missile_pos[0] = next_pos[0]
                    self.missile_pos[1] = next_pos[1]
                    self.missile_pos[2] = max(1.0, next_pos[2])
                    self.missile_vel[2] = max(0, self.missile_vel[2])
            else:
                self.missile_pos = next_pos
            
            if (abs(self.missile_pos[0]) > self.bounds * 2 or 
                abs(self.missile_pos[1]) > self.bounds * 2 or 
                self.missile_pos[2] > self.bounds * 4):
                self.missile_active = False
                print("MISSILE OUT OF BOUNDS")
            
            if self.frame_count % 2 == 0:
                self.missile_path.append(self.missile_pos.copy())
                if len(self.missile_path) > self.max_path_length:
                    self.missile_path.pop(0)
    
    def draw_grid(self):
        """Draw ground grid"""
        glColor4f(0.3, 0.3, 0.3, 0.5)
        glBegin(GL_LINES)
        step = 2000
        for i in range(-10, 11):
            glVertex3f(i * step, -20000, 0)
            glVertex3f(i * step, 20000, 0)
            glVertex3f(-20000, i * step, 0)
            glVertex3f(20000, i * step, 0)
        glEnd()
    
    def draw_sphere(self, pos, radius, color):
        """Draw a sphere"""
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glColor4f(*color)
        
        # Simple sphere approximation
        segments = 8
        for i in range(segments):
            lat0 = np.pi * (-0.5 + float(i) / segments)
            lat1 = np.pi * (-0.5 + float(i + 1) / segments)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(segments + 1):
                lng = 2 * np.pi * float(j) / segments
                x = np.cos(lng)
                y = np.sin(lng)
                
                glVertex3f(radius * x * np.cos(lat0), radius * y * np.cos(lat0), radius * np.sin(lat0))
                glVertex3f(radius * x * np.cos(lat1), radius * y * np.cos(lat1), radius * np.sin(lat1))
            glEnd()
        
        glPopMatrix()
    
    def draw_obj_part(self, obj_name, color, deflection_angle=0.0, rotation_axis=None):
        """Draw a single OBJ object with optional deflection rotation"""
        if obj_name not in self.obj_data or obj_name not in self.obj_centers:
            return
        
        glColor4f(*color)
        # Scale model from meters to match world scale (100x)
        model_scale = 100.0
        center = self.obj_centers[obj_name] * model_scale
        
        # Translate to center, rotate, translate back
        glPushMatrix()
        glTranslatef(center[0], center[1], center[2])
        
        # Apply deflection rotation around specified axis
        if deflection_angle != 0.0 and rotation_axis is not None:
            angle_deg = np.degrees(deflection_angle)
            if rotation_axis == 'y':  # Left/Right canards rotate around Y
                glRotatef(angle_deg, 0, 1, 0)
            elif rotation_axis == 'z':  # Top/Bottom canards rotate around Z
                glRotatef(angle_deg, 0, 0, 1)
        
        glTranslatef(-center[0], -center[1], -center[2])
        
        # Draw wireframe edges with scaling
        glLineWidth(2.0)
        model_scale = 100.0
        for face in self.obj_data[obj_name]:
            glBegin(GL_LINE_LOOP)
            for vertex in face:
                glVertex3f(vertex[0] * model_scale, vertex[1] * model_scale, vertex[2] * model_scale)
            glEnd()
        glLineWidth(1.0)
        
        glPopMatrix()
    
    def draw_aero_forces(self):
        """Draw aerodynamic force vectors on control surfaces along their normals"""
        if not self.missile_active or not hasattr(self, 'canard_forces'):
            return
        
        model_scale = 100.0
        force_scale = 10.0  # Scale forces for visibility
        
        glPushMatrix()
        glTranslatef(self.missile_pos[0], self.missile_pos[1], self.missile_pos[2])
        
        # Apply missile orientation
        if 'Body' in self.obj_centers:
            body_center = self.obj_centers['Body'] * model_scale
            glTranslatef(body_center[0], body_center[1], body_center[2])
        
        yaw, pitch, roll = self.quaternion_to_euler(self.missile_quat)
        glRotatef(np.degrees(yaw), 0, 0, 1)
        glRotatef(np.degrees(pitch), 0, 1, 0)
        glRotatef(np.degrees(roll), 1, 0, 0)
        
        if 'Body' in self.obj_centers:
            body_center = self.obj_centers['Body'] * model_scale
            glTranslatef(-body_center[0], -body_center[1], -body_center[2])
        
        # Draw canard forces (magenta) - along surface normal
        canard_names = ['Can_TOP', 'Can_DOWN', 'Can_LEFT', 'Can_RIGHT']
        canard_axes = ['z', 'z', 'y', 'y']
        
        for i, (obj_name, axis) in enumerate(zip(canard_names, canard_axes)):
            if obj_name in self.obj_centers:
                force_magnitude = np.linalg.norm(self.canard_forces[i])
                if force_magnitude > 0.1:
                    # Get vertex center
                    center = self.obj_centers[obj_name] * model_scale
                    
                    # Get surface normal with deflection applied
                    normal = self.get_surface_normal(obj_name, self.canard_deflection[i], axis)
                    
                    # Force vector along normal, scaled by magnitude
                    force_vector = normal * force_magnitude * force_scale
                    end = center + force_vector
                    
                    glLineWidth(4.0)
                    glColor4f(1.0, 0.0, 1.0, 1.0)  # Magenta
                    glBegin(GL_LINES)
                    glVertex3f(center[0], center[1], center[2])
                    glVertex3f(end[0], end[1], end[2])
                    glEnd()
                    
                    # Draw arrowhead
                    arrow_size = 5.0
                    glPointSize(8.0)
                    glBegin(GL_POINTS)
                    glVertex3f(end[0], end[1], end[2])
                    glEnd()
        
        # Draw fin forces (cyan) - along surface normal
        fin_names = ['Stab_TOP', 'Stab_DOWN', 'Stab_LEFT', 'Stab_RIGHT']
        fin_axes = ['z', 'z', 'y', 'y']
        
        for i, (obj_name, axis) in enumerate(zip(fin_names, fin_axes)):
            if obj_name in self.obj_centers:
                force_magnitude = np.linalg.norm(self.fin_forces[i])
                if force_magnitude > 0.1:
                    # Get vertex center
                    center = self.obj_centers[obj_name] * model_scale
                    
                    # Get surface normal with deflection applied
                    normal = self.get_surface_normal(obj_name, self.fin_deflection[i], axis)
                    
                    # Force vector along normal, scaled by magnitude
                    force_vector = normal * force_magnitude * force_scale
                    end = center + force_vector
                    
                    glLineWidth(4.0)
                    glColor4f(0.0, 1.0, 1.0, 1.0)  # Cyan
                    glBegin(GL_LINES)
                    glVertex3f(center[0], center[1], center[2])
                    glVertex3f(end[0], end[1], end[2])
                    glEnd()
                    
                    # Draw arrowhead
                    glPointSize(8.0)
                    glBegin(GL_POINTS)
                    glVertex3f(end[0], end[1], end[2])
                    glEnd()
        
        glLineWidth(1.0)
        glPointSize(1.0)
        glPopMatrix()
    
    def draw_missile(self):
        """Draw missile with 3D model"""
        glPushMatrix()
        glTranslatef(self.missile_pos[0], self.missile_pos[1], self.missile_pos[2])
        
        # Apply missile orientation - rotate around missile center
        model_scale = 100.0
        if 'Body' in self.obj_centers:
            body_center = self.obj_centers['Body'] * model_scale
            glTranslatef(body_center[0], body_center[1], body_center[2])
        
        # Apply rotation from quaternion (converted to Euler for OpenGL)
        yaw, pitch, roll = self.quaternion_to_euler(self.missile_quat)
        glRotatef(np.degrees(yaw), 0, 0, 1)
        glRotatef(np.degrees(pitch), 0, 1, 0)
        glRotatef(np.degrees(roll), 1, 0, 0)
        
        if 'Body' in self.obj_centers:
            body_center = self.obj_centers['Body'] * model_scale
            glTranslatef(-body_center[0], -body_center[1], -body_center[2])
        
        # Draw body
        if self.obj_data:
            self.draw_obj_part('Body', (0.7, 0.7, 0.7, 1.0))
            
            # Draw canards with deflection (red)
            self.draw_obj_part('Can_TOP', (1.0, 0.2, 0.2, 1.0), self.canard_deflection[0], 'z')
            self.draw_obj_part('Can_DOWN', (1.0, 0.2, 0.2, 1.0), self.canard_deflection[1], 'z')
            self.draw_obj_part('Can_LEFT', (1.0, 0.2, 0.2, 1.0), self.canard_deflection[2], 'y')
            self.draw_obj_part('Can_RIGHT', (1.0, 0.2, 0.2, 1.0), self.canard_deflection[3], 'y')
            
            # Draw fins (blue)
            self.draw_obj_part('Stab_TOP', (0.2, 0.4, 1.0, 1.0), self.fin_deflection[0], 'z')
            self.draw_obj_part('Stab_DOWN', (0.2, 0.4, 1.0, 1.0), self.fin_deflection[1], 'z')
            self.draw_obj_part('Stab_LEFT', (0.2, 0.4, 1.0, 1.0), self.fin_deflection[2], 'y')
            self.draw_obj_part('Stab_RIGHT', (0.2, 0.4, 1.0, 1.0), self.fin_deflection[3], 'y')
        else:
            # Fallback simple missile if OBJ not loaded
            glColor4f(1.0, 0.2, 0.2, 1.0)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(100, 0, 0)
            glEnd()
            
            glColor4f(1.0, 0.0, 0.0, 1.0)
            glBegin(GL_LINES)
            glVertex3f(100, 0, 0)
            glVertex3f(120, 0, 0)
            glEnd()
        
        glPopMatrix()
    
    def draw_path(self, path, color):
        """Draw trajectory path"""
        if len(path) < 2:
            return
        
        glColor4f(*color)
        glBegin(GL_LINE_STRIP)
        for pos in path:
            glVertex3f(pos[0], pos[1], pos[2])
        glEnd()
    
    def draw_line(self, start, end, color, width=1.0):
        """Draw a line"""
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(start[0], start[1], start[2])
        glVertex3f(end[0], end[1], end[2])
        glEnd()
        glLineWidth(1.0)
    
    def render_3d(self):
        """Render 3D scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Setup camera
        if self.follow_missile and self.missile_spawned:
            look_at = self.missile_pos
        else:
            look_at = np.array([0.0, 0.0, 2500.0])
        
        cam_x = look_at[0] + self.camera_distance * np.cos(np.radians(self.camera_angle_h)) * np.cos(np.radians(self.camera_angle_v))
        cam_y = look_at[1] + self.camera_distance * np.sin(np.radians(self.camera_angle_h)) * np.cos(np.radians(self.camera_angle_v))
        cam_z = look_at[2] + self.camera_distance * np.sin(np.radians(self.camera_angle_v))
        
        gluLookAt(cam_x, cam_y, cam_z,
                  look_at[0], look_at[1], look_at[2],
                  0, 0, 1)
        
        # Draw scene
        self.draw_grid()
        
        # Target
        self.draw_sphere(self.target_pos, 80, (0.2, 0.4, 1.0, 1.0))
        self.draw_path(self.target_path, (0.2, 0.4, 1.0, 0.6))
        
        # Missile
        if self.missile_spawned:
            self.draw_missile()
            self.draw_aero_forces()  # Draw force vectors on control surfaces
            self.draw_path(self.missile_path, (1.0, 0.2, 0.2, 0.8))
        
        # Debug lines
        if self.missile_active:
            # Line of sight
            self.draw_line(self.missile_pos, self.target_pos, (0.0, 1.0, 0.0, 0.5), 2.0)
            
            # Velocity vector
            vel_scale = 100
            vel_end = self.missile_pos + self.missile_vel * vel_scale / np.linalg.norm(self.missile_vel) if np.linalg.norm(self.missile_vel) > 1 else self.missile_pos
            self.draw_line(self.missile_pos, vel_end, (1.0, 1.0, 0.0, 0.8), 3.0)
    
    def draw_text(self, text, x, y, color=(255, 255, 255), font=None):
        """Draw 2D text overlay"""
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tobytes(text_surface, "RGBA", True)
        
        glWindowPos2d(x, self.height - y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                     GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def render_hud(self):
        """Render HUD overlay"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        
        # HUD text
        y = 20
        if self.missile_active:
            distance = np.linalg.norm(self.target_pos - self.missile_pos) / 1000.0
            speed_kmh = np.linalg.norm(self.missile_vel) * 3.6
            fuel_left = max(0, self.missile_fuel - self.missile_time)
            alt_km = self.missile_pos[2] / 1000.0
            yaw, pitch, roll = self.quaternion_to_euler(self.missile_quat)
            
            self.draw_text(f"MISSILE ACTIVE", 10, y, (255, 100, 100))
            y += 25
            self.draw_text(f"Distance: {distance:.2f} km | Speed: {speed_kmh:.0f} km/h", 10, y)
            y += 25
            self.draw_text(f"Altitude: {alt_km:.2f} km | Fuel: {fuel_left:.1f}s", 10, y)
            y += 25
            self.draw_text(f"Time: {self.missile_time:.1f}s | Sim Speed: {self.sim_speed:.1f}x", 10, y)
            y += 25
            self.draw_text(f"Pitch: {np.degrees(pitch):+.1f}° | Yaw: {np.degrees(yaw):+.1f}° | Roll: {np.degrees(roll):+.1f}°", 10, y, (150, 150, 150))
        elif self.missile_spawned:
            distance = np.linalg.norm(self.target_pos - self.missile_pos) / 1000.0
            self.draw_text(f"MISSILE READY - Distance: {distance:.2f} km", 10, y, (100, 255, 100))
            y += 25
            self.draw_text(f"Press SPACE to launch", 10, y)
        else:
            self.draw_text(f"Press SPACE to spawn missile", 10, y)
        
        # FPS counter
        self.draw_text(f"FPS: {self.fps:.0f}", self.width - 100, 20, (200, 200, 200))
        
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def handle_events(self):
        """Handle input events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_SPACE:
                    if not self.missile_spawned:
                        self.missile_spawned = True
                        self.missile_pos = np.array([0.0, 0.0, 50.0])
                        self.missile_vel = np.array([0.0, 0.0, 0.0])
                        self.missile_quat = self.euler_to_quaternion(0.0, -np.pi / 2.0, 0.0)
                        self.missile_path = [self.missile_pos.copy()]
                        print("MISSILE SPAWNED")
                    elif not self.missile_active:
                        self.missile_vel = np.array([0.0, 0.0, 100.0])
                        self.missile_active = True
                        self.missile_time = 0.0
                        print("MISSILE LAUNCHED")
                elif event.key == K_r:
                    self.missile_active = False
                    self.missile_spawned = True
                    self.missile_pos = np.array([0.0, 0.0, 50.0])
                    self.missile_vel = np.array([0.0, 0.0, 0.0])
                    self.missile_time = 0.0
                    self.missile_quat = self.euler_to_quaternion(0.0, -np.pi / 2.0, 0.0)
                    self.angular_velocity = np.array([0.0, 0.0, 0.0])
                    self.missile_path = [self.missile_pos.copy()]
                    print("RESET")
                elif event.key == K_f:
                    self.follow_missile = not self.follow_missile
                    print(f"Follow missile: {self.follow_missile}")
                elif event.key == K_d:
                    self.debug_enabled = not self.debug_enabled
                    print(f"Debug output: {'ON' if self.debug_enabled else 'OFF'}")
                elif event.key == K_EQUALS or event.key == K_PLUS:
                    self.sim_speed = min(10.0, self.sim_speed + 0.5)
                    print(f"Sim speed: {self.sim_speed:.1f}x")
                elif event.key == K_MINUS:
                    self.sim_speed = max(0.1, self.sim_speed - 0.5)
                    print(f"Sim speed: {self.sim_speed:.1f}x")
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_down = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    self.camera_distance = max(1000, self.camera_distance - 1000)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(50000, self.camera_distance + 1000)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            
            elif event.type == MOUSEMOTION:
                if self.mouse_down:
                    current_pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = current_pos[0] - self.last_mouse_pos[0]
                        dy = current_pos[1] - self.last_mouse_pos[1]
                        self.camera_angle_h -= dx * 0.3  # Fixed: reversed for correct rotation
                        self.camera_angle_v += dy * 0.3
                        self.camera_angle_v = np.clip(self.camera_angle_v, -89, 89)
                    self.last_mouse_pos = current_pos
        
        return True
    
    def run(self):
        """Main game loop"""
        running = True
        self.start_time = pygame.time.get_ticks() / 1000.0
        
        while running:
            # Auto-launch missile after delay
            if not self.missile_launched:
                current_time = pygame.time.get_ticks() / 1000.0
                if (current_time - self.start_time) > self.auto_launch_delay:
                    self.missile_launched = True
                    print("\n=== AUTO-LAUNCHING MISSILE ===")
                    self.missile_spawned = True
                    self.missile_pos = np.array([0.0, 0.0, 0.0])
                    self.missile_vel = np.array([0.0, 0.0, 0.0])
                    self.missile_time = 0.0
                    self.missile_quat = self.euler_to_quaternion(0.0, -np.pi / 2.0, 0.0)
                    self.angular_velocity = np.array([0.0, 0.0, 0.0])
                    self.missile_path = [self.missile_pos.copy()]
            
            # Handle events
            running = self.handle_events()
            
            # Update physics
            self.update_physics()
            
            # Render
            self.render_3d()
            self.render_hud()
            
            # Swap buffers
            pygame.display.flip()
            
            # Maintain FPS
            self.clock.tick(60)
            self.fps = self.clock.get_fps()
            self.frame_count += 1
        
        pygame.quit()

if __name__ == "__main__":
    sim = MissileSimulation()
    sim.run()

import sys, random
from math import *

import pygame
from pygame.locals import *
import numpy as np
from numpy import array as vec

# ==================================================================================
# Constants
# ==================================================================================

# Display / timing
SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600
SCREEN_SIZE   = vec([SCREEN_WIDTH, SCREEN_HEIGHT])
FPS           = 60

BLACK = (0, 0, 0)

# Example aero constants (currently not used but kept for future use)
AIR_DENSITY     = 1.225    # kg/m^3
FIN_AREA        = 0.02     # m^2
CANARD_AREA     = 0.01     # m^2
CL_ALPHA_FIN    = 4.0
CL_ALPHA_CANARD = 4.0

# Gravity used in physics
GRAVITY = vec([0.0, -9.81])  # world Y up (subtracted in update => same effect as original)

# ==================================================================================
# Utility functions
# ==================================================================================

def dist(p):
    """Euclidean length of a 2D vector."""
    return sqrt(p[0] ** 2 + p[1] ** 2)


def rotate_point(p, angle):
    """
    Rotate a point p by 'angle' around origin.
    NOTE: keeps the original (somewhat unusual) sin/cos ordering
    so that behavior remains identical.
    """
    length = dist(p)
    base_angle = atan2(p[1], p[0])
    return vec([sin(base_angle + angle), cos(base_angle + angle)]) * length


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + pi) % (2.0 * pi) - pi

def norm(v):
    return v / dist(v)

def moment_about_center(O, F):
    Ox, Oy = O
    Fx, Fy = F
    Mz = Ox * Fy - Oy * Fx
    return Mz

def calculate_lift(aoa, vel):
    """
    Flat-plate-like lift model with simple stall.
    Returns a lift magnitude (scalar) aligned with
    the local lift direction in update().
    """
    # 1) Wrap AoA to [-pi, pi]
    aoa_rel = normalize_angle(aoa)

    # 2) Effective AoA in [-pi/2, pi/2]
    aoa_eff = (aoa_rel + 0.5 * pi) % pi - 0.5 * pi  # [-pi/2, +pi/2]

    V = dist(vel)

    # Parameters (unchanged)
    CL_alpha    = 5.5
    alpha_stall = radians(15.0)
    alpha_max   = radians(30.0)
    k_lift      = 0.03

    abs_a = abs(aoa_eff)

    # 3) Stall model on effective AoA
    if abs_a <= alpha_stall:
        CL = CL_alpha * aoa_eff
    else:
        t = (abs_a - alpha_stall) / (alpha_max - alpha_stall)
        t = max(0.0, min(1.0, t))
        CL_stall = CL_alpha * alpha_stall
        CL = copysign((1.0 - t) * CL_stall, aoa_eff)

    # 4) Optional sign flip depending on flow (kept commented out)
    # sign_flow = 1.0 if cos(aoa_rel) >= 0.0 else -1.0
    # CL *= sign_flow

    return k_lift * CL * V * V


# ==================================================================================
# Tarrget
# ==================================================================================
class Wing:
    def __init__(self, surface, cl_a, cd0, ar=6.0, e=0.75, rho=1.225):
        self.area = surface # S [m^2]
        self.cl_alpha = cl_a # dCL/d(alpha) [1/rad]
        self.cd0 = cd0 # parasitic drag coefficient
        self.aspect_ratio = ar # AR, used for induced drag
        self.e = e # Oswald efficiency factor
        self.rho = rho # air density [kg/m^3]
        self.cl_max = 4.25
        self.cd_alpha = 0.175

canard = Wing(0.012, 4.0, 0.01, 2, 0.7)
fin = Wing(0.025, 4.5, 0.015, 2.5, 0.8)

def compute_lift_drag(vel, wing_angle, wing):
    vx, vy = vel
    V = dist(vel)
    if V < 1e-6:
        return vec([0.0, 0.0]), vec([0.0, 0.0])

    flow_angle = atan2(vy, vx)
    alpha = wing_angle - flow_angle  # [rad]

    # wrap AoA to [-pi, pi] if needed
    while alpha > 3.14159265:
        alpha -= 2.0 * 3.14159265
    while alpha < -3.14159265:
        alpha += 2.0 * 3.14159265

    q = 0.5 * wing.rho * V * V

    # Nonlinear coefficients
    # cl roughly ~ sin(2α): 0 at 0°, sign changes with α, max at 45°
    cl = wing.cl_max * sin(2.0 * alpha)

    # drag: base + induced + extra with AoA
    cd = wing.cd0 + wing.cd_alpha * (sin(alpha) ** 2)

    L = q * wing.area * cl
    D = q * wing.area * cd

    v_hat = vel / V

    drag_vec = -D * v_hat
    lift_vec = vec([L * -v_hat[1], L * v_hat[0]])

    return lift_vec, drag_vec



class Target:
    def __init__(self, position):
        self.pos = vec(position)
        self.vel = vec([700/3.6, 0.0])

    def draw(self):
        pygame.draw.circle(
            screen,
            (0, 255, 0),
            cam.world_to_screen(self.pos),
            10
        )
    
    def update(self, dt, t):
        #self.vel += vec([cos(t)*30, 0])
        self.pos += self.vel * dt


# ==================================================================================
# Camera
# ==================================================================================

class Camera:
    def __init__(self):
        # center = point in WORLD coordinates that is at the center of the screen
        self.center = vec([0.0, 0.0])
        self.zoom   = 50.0

        # dragging
        self.dragging        = False
        self.drag_start_mouse  = vec([0.0, 0.0])
        self.drag_start_center = vec([0.0, 0.0])

    # ------------------------------------------------------------------ transforms

    def world_to_screen(self, p):
        """Convert world position (2D) -> screen position (2D)."""
        p = vec(p) * vec([1, -1])
        return (p - self.center) * self.zoom + SCREEN_SIZE / 2.0

    def screen_to_world(self, s):
        """Convert screen position (2D) -> world position (2D)."""
        s = vec(s) * vec([1, -1])
        return (s - SCREEN_SIZE / 2.0) / self.zoom + self.center

    def world_to_cam_rect(self, rect):
        """Convert a world rect [x,y,w,h] to a pygame.Rect on screen."""
        x, y, w, h = rect
        screen_pos = self.world_to_screen(vec([x, y]) * vec([1, -1]))
        return pygame.Rect(screen_pos[0], screen_pos[1], w * self.zoom, h * self.zoom)

    def world_to_cam_poly(self, poly):
        """Convert a list of world vertices to screen vertices in-place."""
        for i, vert in enumerate(poly):
            poly[i] = (vert * vec([1, -1]) - self.center) * self.zoom + SCREEN_SIZE / 2.0
        return poly

    # ------------------------------------------------------------------ dragging

    def start_drag(self, mouse_pos):
        self.dragging = True
        self.drag_start_mouse  = vec(mouse_pos)
        self.drag_start_center = self.center.copy()

    def update_drag(self, mouse_pos):
        if not self.dragging:
            return
        mouse_pos   = vec(mouse_pos)
        delta_screen = mouse_pos - self.drag_start_mouse
        # divide by zoom so drag speed doesn't depend on zoom level
        self.center = self.drag_start_center - delta_screen / self.zoom

    def stop_drag(self):
        self.dragging = False

    # ------------------------------------------------------------------ zoom

    def zoom_at(self, mouse_pos, zoom_factor):
        """
        Zoom relative to the point under mouse_pos (screen coords).
        zoom_factor > 1 => zoom in, < 1 => zoom out
        """
        mouse_pos = vec(mouse_pos)
        old_zoom  = self.zoom
        new_zoom  = self.zoom * zoom_factor

        # clamp zoom
        new_zoom = max(0.1, min(new_zoom, 50.0))

        # world position under the cursor BEFORE zoom
        world_before = self.screen_to_world(mouse_pos)

        self.zoom = new_zoom

        # adjust center so world_before still maps to same screen position
        self.center = world_before - (mouse_pos - SCREEN_SIZE / 2.0) / self.zoom


# ==================================================================================
# Missile simulation
# ==================================================================================

class Particle:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.life = 5.0
    
    def update(self, dt):
        self.life -= dt
        if self.life < 0: return False
        self.vel *= 0.99
        self.pos += self.vel * dt
        return True

class MissileSimulation:
    def __init__(self):
        self.pos   = vec([0.0, 0.0])
        self.vel   = vec([0.0, 0.0])
        self.accel = vec([0.0, 0.0])
        self.mass  = 85.5

        self.max_thrust = 25000.0
        self.fuel_time  = 3.0

        self.angle     = pi / 2
        self.angle_vel = 0.0
        self.moment    = 0.0
        self.inertia   = 65.0

        self.canard_angle = 0.0
        self.canard_force = 5.0
        self.fin_force    = 5.0
        self.g = 0.0

        self.debug_force = True

        self.particles = []

    # ------------------------------------------------------------------ drawing

    def _build_body_mesh(self):
        """Return body mesh vertices in local coords."""
        return [
            vec([0.0635,  1.5]),
            vec([-0.0635, 1.5]),
            vec([-0.0635, -1.5]),
            vec([0.0635, -1.5]),
        ]

    def _build_fin_local(self):
        """Return fin endpoints in local coords (before offset)."""
        return [vec([0.0, -0.4]), vec([0.0, 0.4])]

    def _build_canard_local(self):
        """Return canard endpoints in local coords (before rotation)."""
        return [vec([0.0, 0.2]), vec([0.0, -0.2])]

    def draw(self):
        # Body
        mesh_local = self._build_body_mesh()
        mesh_world = [rotate_point(v, self.angle) + self.pos for v in mesh_local]

        # Fin (fixed orientation on body)
        fin_pos_local = vec([0.0, -1.0])
        fin_local     = self._build_fin_local()
        fin_world     = [
            rotate_point(p + fin_pos_local, self.angle) + self.pos
            for p in fin_local
        ]

        # Canard (rotating control surface)
        canard_pos_local = vec([0.0, 1.1])
        canard_local     = self._build_canard_local()
        canard_world     = [
            rotate_point(
                rotate_point(p, self.canard_angle + pi / 2.0) + canard_pos_local,
                self.angle
            ) + self.pos
            for p in canard_local
        ]

        # Draw body and surfaces
        pygame.draw.polygon(screen, (255, 255, 255), cam.world_to_cam_poly(mesh_world))
        pygame.draw.line(
            screen, (255, 0, 0),
            cam.world_to_screen(fin_world[0]),
            cam.world_to_screen(fin_world[1]),
            3
        )
        pygame.draw.line(
            screen, (255, 128, 0),
            cam.world_to_screen(canard_world[0]),
            cam.world_to_screen(canard_world[1]),
            3
        )

        for particle in self.particles:
            screen.set_at(cam.world_to_screen(particle.pos), (255, 255, 255))

        # Debug forces
        if self.debug_force:
            # Canard lift vector
            if False:
                canard_pos    = canard_pos_local
                start_canard  = rotate_point(canard_pos, self.angle) + self.pos
                end_canard    = start_canard + rotate_point(vec([-self.canard_force, 0.0]),
                                                            self.angle - self.canard_angle)*0.0001
                pygame.draw.line(
                    screen, (128, 128, 128),
                    cam.world_to_screen(start_canard),
                    cam.world_to_screen(end_canard),
                    3,
                )

                # Fin lift vector
                fin_pos       = fin_pos_local
                start_fin     = rotate_point(fin_pos, self.angle) + self.pos
                end_fin       = start_fin + rotate_point(vec([-self.fin_force, 0.0]), self.angle)*0.0001
                pygame.draw.line(
                    screen, (128, 128, 128),
                    cam.world_to_screen(start_fin),
                    cam.world_to_screen(end_fin),
                    3,
                )

            # Velocity vector
            start_vel = self.pos
            end_vel   = start_vel + self.vel * 0.001
            pygame.draw.line(
                screen, (128, 128, 128),
                cam.world_to_screen(start_vel),
                cam.world_to_screen(end_vel),
                3,
            )

    # ------------------------------------------------------------------ physics

    def _apply_thrust_and_gravity(self):
        # Fuel
        self.fuel_time = max(self.fuel_time, 0.0)

        # Gravity (unchanged sign/operation)
        self.accel -= GRAVITY

        # Thrust
        if self.fuel_time != 0:
            for i in range(5):
                angle = self.angle - pi/2.0 + (random.random()-0.5)*0.3
                vel = vec([sin(angle), cos(angle)]) * 5 + random.random()
                self.particles.append(Particle(vec(self.pos), vel+self.vel))

            thrust_dir = vec([cos(-self.angle), sin(-self.angle)])
            self.accel += thrust_dir * self.max_thrust / self.mass

    def _apply_fin_forces(self):
        lift, drag = compute_lift_drag(self.vel, self.angle-pi, canard)
        
        fin_pos_local = vec([0.0, -1.0])

        # Lift magnitude (unchanged formula)
        self.fin_force = lift

        # Acceleration from canard lift (same direction as fin lift)
        
        self.accel -= self.fin_force / self.mass*0

        # Moment from canard
        offset = rotate_point(fin_pos_local, self.angle)
        self.moment -= moment_about_center(offset, self.fin_force)

    def _apply_canard_forces(self):
        lift, drag = compute_lift_drag(self.vel, normalize_angle(self.angle+self.canard_angle-pi), canard)
        
        canard_pos_local = vec([0.0, -1.0])

        # Lift magnitude (unchanged formula)
        self.canard_force = lift

        self.accel + self.canard_force / self.mass*0

        # Moment from canard
        offset = rotate_point(canard_pos_local, self.angle)
        self.moment += moment_about_center(offset, self.canard_force)

    def _integrate_linear(self, dt):
        self.vel   += self.accel * dt
        self.pos   += self.vel * dt
        self.g = dist(self.accel/9.81)
        self.accel  = vec([0.0, 0.0])

    def _integrate_angular(self, dt):
        angular_accel = self.moment / self.inertia
        self.angle_vel += angular_accel * dt
        self.angle     += self.angle_vel * dt
        self.moment     = 0.0

    def _guidance_equation(self, target, debug_surface):
        target_vec = target.pos - self.pos
        D = dist(target_vec)
        T = D/(dist(self.vel)+1)

        new_target_pos = target.pos + target.vel * T
        target_vec = new_target_pos - self.pos

        pygame.draw.line(debug_surface, (0, 255, 0), cam.world_to_screen(target.pos), cam.world_to_screen(target.pos+target_vec))
        pygame.draw.line(debug_surface, (255, 0, 0), cam.world_to_screen(self.pos), cam.world_to_screen(target_vec+self.pos), 3)
        target_angle = atan2(target_vec[0], target_vec[1])
        
        text = f"g: {round(self.g, 1)}\nspeed: {round(dist(self.vel)*3.6)} km/h"
        text_surface = debug_font.render(text, True, (255, 255, 255))


        debug_surface.blit(text_surface, (10, 10))

        # Calculate angle error
        angle_error = normalize_angle(target_angle - (self.angle+pi/2.0))
        #self.canard_angle = -min(max(angle_error, -0.1), 0.1) * 0
    
    def update(self, dt, debug_surface):
        # fuel countdown
        self.fuel_time -= dt
        self._guidance_equation(target, debug_surface)
        self._apply_thrust_and_gravity()
        self._apply_fin_forces()
        self._apply_canard_forces()
        self._integrate_linear(dt)
        self._integrate_angular(dt)
        
        while 200 < len(self.particles):
            self.particles.pop(0)

        for particle in self.particles:
            particle.update(dt)


# ==================================================================================
# Grid drawing
# ==================================================================================

def draw_grid(screen, cam, spacing=1.0, color=(60, 60, 60), axis_color=(200, 60, 60)):
    """
    Draw a background grid in WORLD space with given `spacing` (world units).
    The camera handles world->screen conversion so lines remain correct when
    panning/zooming.
    """
    # World coordinates of the screen corners
    top_left     = cam.screen_to_world(vec([0.0, 0.0]))
    bottom_right = cam.screen_to_world(SCREEN_SIZE)

    xmin = min(top_left[0], bottom_right[0])
    xmax = max(top_left[0], bottom_right[0])
    ymin = min(top_left[1], bottom_right[1])
    ymax = max(top_left[1], bottom_right[1])

    # expand a bit so edges are covered
    pad = spacing * 1
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    # Vertical lines
    start_x = floor(xmin / spacing) * spacing
    x = start_x
    while x <= xmax:
        p1 = cam.world_to_screen(vec([x, ymin]))
        p2 = cam.world_to_screen(vec([x, ymax]))
        line_color = axis_color if abs(x) < 1e-9 else color
        pygame.draw.line(screen, line_color, (p1[0], p1[1]), (p2[0], p2[1]), 1)
        x += spacing

    # Horizontal lines
    start_y = floor(ymin / spacing) * spacing
    y = start_y
    while y <= ymax:
        p1 = cam.world_to_screen(vec([xmin, y]))
        p2 = cam.world_to_screen(vec([xmax, y]))
        line_color = axis_color if abs(y) < 1e-9 else color
        pygame.draw.line(screen, line_color, (p1[0], p1[1]), (p2[0], p2[1]), 1)
        y += spacing


# ==================================================================================
# Main loop
# ==================================================================================

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
debug_surface = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Missile Simulation 2D")

debug_font = pygame.font.SysFont('Arial', 11, bold=True)

clock  = pygame.time.Clock()
cam    = Camera()
missle = MissileSimulation()
target = Target([-3000.0, -3000.0])
running = True

t = 0.0

if __name__ == "__main__":
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = 1/FPS
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # right mouse button drag
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # right click
                    cam.start_drag(event.pos)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    cam.stop_drag()

            # zoom from mouse cursor
            if event.type == pygame.MOUSEWHEEL:
                print(event.precise_y)
                cam.zoom_at(pygame.mouse.get_pos(), event.precise_y * 0.1 + 1)

        # update dragging state
        if cam.dragging and pygame.mouse.get_pressed()[2]:
            cam.update_drag(pygame.mouse.get_pos())
        else:
            cam.dragging = False

        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            missle.canard_angle += dt*0.1
        if keys[K_RIGHT]:
            missle.canard_angle -= dt*0.1
        debug_surface.fill(BLACK)
        target.update(dt, t)
        missle.update(dt, debug_surface)
        cam.center = missle.pos

        screen.fill(BLACK)
        screen.blit(debug_surface, (0, 0))

        # Optionally draw grid:
        draw_grid(screen, cam, spacing=50)

        missle.draw()
        target.draw()

        pygame.display.flip()

    pygame.quit()
    sys.exit()

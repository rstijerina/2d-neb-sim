__author__ = 'Sal'

"""

    This program uses Pygame to display the interaction of particles in a gas cloud. I use the properties of the
    Minimum Mass Solar Nebula (MMSN) model for an initial density profile of 1700 g/cm^3.

    The program runs through an N-body simulation of newtonian particle interactions, conserving momentum, using the
    Adaptive Runge-Kutta method.

    Can easily change the gravitational constant, number of particles, density of material, and whether or not a star
    exists to experiment with results.

    Displays time, which should be compared with the timescale for gravitational collapse:

        t_collapse ~ 1/sqrt(G*rho)


    ** Computational time constrains the real-time output of the program, as the rka method is time-intensive.

    ** Future work would include creating videos of runs of a more appropriate number of particles.

"""
import numpy as np
import random
import pygame

density = 1.7e6     # density in kg/m^2
G = 6.67e-11        # Grav. constant m^3/(kg^2 s^2)
# G = 1e3
AU = 1.496e11       # AU in meters
yr = 3.154e7        # year in seconds

star = False        # Whether or not to include a solar-mass star in the center of the system

number_of_particles = 20

t_step = 5       # timestep in seconds

Width = 480      # Dimensions of the Pygame screen
Height = 360


class Particle:
    def __init__(self):
        self.x = float(random.randint(Width/4, 3*Width/4))
        self.y = float(random.randint(Width/4, 3*Height/4))
        # self.vx = float(random.randint(0, 25))
        # self.vy = float(random.randint(0, 25))
        self.vx = 0.
        self.vy = 0.
        self.radius = 2.     # radius in meters
        self.mass = density*4.*np.pi*self.radius**3 / 3.
        self.time = 0.
        self.tau = .01

        self.collision = False

    def rk4(self, x, t, tau):
        #  Runge-Kutta integrator (4th order)
        # Input arguments -
        #   x = current value of dependent variable
        #   t = independent variable (usually time)
        #   tau = step size (usually timestep)
        #   derivsRK = right hand side of the ODE; derivsRK is the
        #             name of the function which returns dx/dt
        #             Calling format derivsRK(x,t).
        # Output arguments -
        #   xout = new value of x after a step of size tau

        # x = np.array([self.x, self.y, self.vx, self.vy])

        xout = np.array([0.,0.,0.,0.])
        half_tau = 0.5*tau
        F1 = self.gravrk(x, t)

        t_half = t + half_tau
        xtemp = x + half_tau*F1
        F2 = self.gravrk(xtemp, t_half)
        xtemp = x + half_tau*F2
        F3 = self.gravrk(xtemp, t_half)
        t_full = t + tau
        xtemp = x + tau*F3
        F4 = self.gravrk(xtemp, t_full)

        xout[0] = x[0] + tau/6.*(F1[0] + F4[0] + 2.*(F2[0]+F3[0]))
        xout[1] = x[1] + tau/6.*(F1[1] + F4[1] + 2.*(F2[1]+F3[1]))
        xout[2] = x[2] + tau/6.*(F1[2] + F4[2] + 2.*(F2[2]+F3[2]))
        xout[3] = x[3] + tau/6.*(F1[3] + F4[3] + 2.*(F2[3]+F3[3]))

        # self.x_new = xout[0]
        # self.y_new = xout[1]
        # self.vx_new = xout[2]
        # self.vy_new = xout[3]
        # self.tau_new = tau
        # self.time_new = t
        return xout

    def rka(self, err, time, tau):

        # Adaptive Runge-Kutta routine
        # Inputs
        #   x          Current value of the dependent variable
        #   t          Independent variable (usually time)
        #   tau        Step size (usually time step)
        #   err        Desired fractional local truncation error
        #   derivsRK   Right hand side of the ODE; derivsRK is the
        #              name of the function which returns dx/dt
        #              Calling format derivsRK(x,t).
        # Outputs
        #   xSmall     New value of the dependent variable
        #   t          New value of the independent variable
        #   tau        Suggested step size for next call to rka

        # Set initial variables
        tSave = time
        xSave = [self.x, self.y, self.vx, self.vy]    # Save initial values
        # tau = self.tau
        safe1 = .9;  safe2 = 4.  # Safety factors
        eps = np.spacing(1)  # smallest value

        # Loop over maximum number of attempts to satisfy error bound
        maxTry = 100

        for iTry in range(1, maxTry):

            #  Take the two small time steps
            half_tau = 0.5 * tau
            xTemp = self.rk4(xSave, tSave, half_tau)
            t = tSave + half_tau
            xSmall = self.rk4(xTemp, t, half_tau)

            # Take the single big time step
            t = tSave + tau
            xBig = self.rk4(xSave, tSave, tau)

            # Compute the estimated truncation error
            scale = err * (np.abs(xSmall) + np.abs(xBig))/2.
            xDiff = xSmall - xBig
            errorRatio = np.max([np.abs(xDiff)/(scale + eps)])
            # print(errorRatio)

            # print safe1,tau,errorRatio

            # Estimate news tau value (including safety factors)
            tau_old = tau

            tau = safe1*tau_old*errorRatio**(-0.20)
            tau = np.max([tau, tau_old/safe2])
            tau = np.min([tau, safe2*tau_old])

            # If error is acceptable, return computed values
            if errorRatio < 1:

                self.x_new = xSmall[0]
                self.y_new = xSmall[1]
                self.vx_new = xSmall[2]
                self.vy_new = xSmall[3]
                self.tau_new = tau
                self.time_new = t
                return xSmall, t, tau

        # Issue error message if error bound never satisfied
        print('ERROR: Adaptive Runge-Kutta routine failed')
        return

    def gravrk(self, s, t):
        #  Returns right-hand side of Kepler ODE; used by Runge-Kutta routines
        #  Inputs
        #    s      State vector [r(1) r(2) v(1) v(2)]
        #    t      Time (not used)
        #  Output
        #    deriv  Derivatives [dr(1)/dt dr(2)/dt dv(1)/dt dv(2)/dt]

        # Compute acceleration
        r = np.array([s[0], s[1]])  # Unravel the vector s into position and velocity
        v = np.array([s[2], s[3]])
        # accel = np.array(0, 0)
        accel = [0., 0.]
        for pp in particles:
            if pp is self or pp.collision:
                continue  # ignore ourselves and merged planets
            rp = np.array([pp.x, pp.y])

            accel -= G*pp.mass*(r - rp)/np.linalg.norm(r-rp)**3

        derivs = np.array([v[0], v[1], accel[0], accel[1]])

        return derivs

    def update(self):
        self.x = self.x_new
        self.y = self.y_new
        self.vx = self.vx_new
        self.vy = self.vy_new
        # self.tau = self.tau_new
        # self.time = self.time_new
        self.time += self.time*self.tau
        self.radius = (3*self.mass/(4*np.pi*density))**(1/3)


def check_for_collision(particle1, particle2):
    dist = np.sqrt((particle1.x - particle2.x)**2 + (particle1.y - particle2.y)**2)

    if dist <= (particle1.radius + particle2.radius):
        # print('p1(x,y) = (%d,%d) and p2(x,y) = (%d,%d)' % (particle1.x, particle1.y, particle2.x, particle2.y))
        # print('colliding, dist = %d' % dist)
        # print('p1 rad = %d and p2 rad = %d' % (particle1.radius, particle2.radius))
        if particle1.mass > particle2.mass:
            particle1.mass += particle2.mass
            particle2.collision = True
            particle1.vx = (particle1.vx*particle1.mass + particle2.vx*particle2.mass)/(particle1.mass + particle2.mass)
            particle1.vy = (particle1.vy*particle1.mass + particle2.vy*particle2.mass)/(particle1.mass + particle2.mass)
        else:
            particle2.mass += particle1.mass
            particle1.collision = True
            particle2.vx = (particle1.vx*particle1.mass + particle2.vx*particle2.mass)/(particle1.mass + particle2.mass)
            particle2.vy = (particle1.vy*particle1.mass + particle2.vy*particle2.mass)/(particle1.mass + particle2.mass)


pygame.init()

screen = pygame.display.set_mode((Width, Height))

global particles
particles = []

for p in range(0, number_of_particles-1):

    if p == 0:
        sun = Particle()
        sun.x = Width/2.
        sun.y = Height/2.
        sun.vx = 0.
        sun.vy = 0.
        sun.mass = 1.98e30      # mass of sun in kg
        # sun.mass = 1e3
        sun.radius = 50.
        # sun.radius = 6.95e8     # radius of sun in m
        # sun.radius = 50
        # particles.append(sun)
        if star:
            particles.append(sun)
        else:
            particles.append(Particle())
    else:
        particles.append(Particle())

calculate = True

adaptErr = 1.e-4     # Error parameter used by adaptive Runge-Kutta

time = 0.

for pct1 in particles:
    if pct1.collision:
        continue
    for pct2 in particles:
        if (pct1 == pct2) or (pct2.collision):
            continue
        else:
            check_for_collision(pct1, pct2)

while calculate:
    time += t_step

    font = pygame.font.Font(None, 25)
    text = font.render(str('time=%.2f seconds' % time), 1, (255, 255, 255))
    screen.blit(text, (720, 640))

    pygame.display.flip()
    screen.fill((0, 0, 0))

    for p in particles:
        if p.collision != True:
            size = p.radius
            if size < 2:
                pygame.draw.circle(screen, (255, 255, 255), (int(p.x), int(p.y)), 2, 0)
            elif (size > 15) or p is sun:
                pygame.draw.circle(screen, (255, 255, 0), (int(p.x), int(p.y)), 15, 0)
            else:
                pygame.draw.circle(screen, (255, 255, 255), (int(p.x), int(p.y)), int(size), 0)

    for p in particles:
        p.rka(adaptErr, time, t_step)
        # p.rk4(time, tau)

    for p in particles:
        p.update()
        if p is sun:
            p.radius = 15

    for pct1 in particles:
        if pct1.collision == True:
            continue
        for pct2 in particles:
            if (pct1 == pct2) or (pct2.collision == True):
                continue
            else:
                check_for_collision(pct1, pct2)

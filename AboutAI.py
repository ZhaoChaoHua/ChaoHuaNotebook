import pyglet.gl as gl
import numpy as np
import numpy.matlib
import matplotlib.colors as mcol
import pyglet
import time
import pandas as pd

# O: Object   e.g. vtsO: vertexes in Object Coordinate System
# W: World    e.g. vtsW: vertexes in World Coordinate System
gravity = 9.81
w = 1080  # Screen width
h = 720  # Screen height
lw = 2.0  # Line width

config = gl.Config(sample_buffers=1, samples=8)  # Anti-aliasing
window = pyglet.window.Window(w, h, config=config)

# Angle to rotation matrix
def rad2rm(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.mat([[c, -s], [s, c]])

# Value remap
def remap(x, old_min, old_max, new_min=0., new_max=1.):
    old_mean = .5*(old_max+old_min)
    old_range = old_max - old_min
    new_mean = .5*(new_max+new_min)
    new_range = new_max - new_min
    y = (x-old_mean)*new_range/old_range + new_mean
    return y


class Camera:
    def __init__(self, scale=200.0, theta=0.0, x=0.0, y=0.0):
        self.scale = scale  # Zoom factor: Pixels Per Meter
        self.theta = theta  # Camera roll angle
        self.trans = np.mat([x, y]).T  # Camera position
        self.rotmat = rad2rm(theta)  # Rotation matrix
        self.offset = np.mat([w / 2, h / 2]).T  # Camera offset

    def setPosition(self, x, y):
        self.trans[0, 0] = x
        self.trans[1, 0] = y

    def setX(self, x):
        self.trans[0, 0] = x

    def setY(self, y):
        self.trans[1, 0] = y

    def getX(self):
        return self.trans[0, 0]

    def getY(self):
        return self.trans[1, 0]

    def setDirectionDeg(self, theta):
        self.theta = np.deg2rad(theta)
        self.rotmat = rad2rm(self.theta)

    def setDirectionRad(self, theta):
        self.theta = theta
        self.rotmat = rad2rm(theta)

    def project(self, vts):
        return self.rotmat.T * vts * self.scale - self.trans * self.scale + self.offset


class Element:
    def __init__(self, num_vts=1, x=0.0, y=0.0, camera=None):
        self.num_vts = num_vts  # Vertexes number
        self.vtsO = np.matlib.zeros((2, num_vts))  # Vertexes represented in Object Coordinate System
        self.pivotO = np.matlib.zeros((2, 1))  # Center of rotation represented in Object Coordinate System
        self.vtsW = np.matlib.zeros((2, num_vts))  # Vertexes represented in World Coordinate System
        self.translation = np.mat([[x], [y]])  # Offset in World Coordinate System
        self.rotMat = np.matlib.eye(2)  # Rotation Matrix of Element

        self.camera = camera  # Camera if define
        self.vtsC = np.matlib.zeros((2, num_vts))  # Vertexes represent in Camera Coordinate System

    # Set Center of Rotation represented in Object Coordinate System
    def setPivotO(self, x, y):
        self.pivotO[0] = x
        self.pivotO[1] = y

    # Rotate Element around P(x,y) in World Coordinate System
    def rotateE(self, theta, x=0, y=0):
        rm = rad2rm(theta)
        t = np.mat([[x], [y]])
        self.vtsW = rm * (self.vtsW - t) + t

    # Rotate Element around pivot
    def rotateO(self, theta):
        rm = rad2rm(theta)
        t = self.translation + self.pivotO
        self.vtsW = rm * (self.vtsW - t) + t

    # Move Element
    def translate(self, dx, dy):
        t = np.mat([[dx], [dy]])
        self.vtsW += t

    def setPosition(self, x=0.0, y=0.0):
        self.translation = np.mat([[x], [y]])
        self.vtsW = self.rotMat * (self.vtsO - self.pivotO) + self.translation

    def setDirectionRad(self, theta=0.0):
        self.rotMat = rad2rm(theta)
        self.vtsW = self.rotMat * (self.vtsO - self.pivotO) + self.translation

    def setDirectionDeg(self, theta=0.0):
        self.rotMat = rad2rm(np.deg2rad(theta))
        self.vtsW = self.rotMat * (self.vtsO - self.pivotO) + self.translation

    def setTranslation(self, x, y):  # This function does not refresh position or orientation of element
        self.translation[0] = x
        self.translation[1] = y

    def setRotMat(self, r00, r01, r10, r11):
        self.rotMat[0, 0] = r00
        self.rotMat[0, 1] = r01
        self.rotMat[1, 0] = r10
        self.rotMat[1, 1] = r11

    def setRotMatRad(self, theta):  # This function does not refresh position or orientation of element
        self.rotMat = rad2rm(theta)

    def setRotMatDeg(self, theta):
        self.rotMat = rad2rm(np.deg2rad(theta))

    def setRotMatSinCos(self, sin=0.0, cos=1.0):
        s = sin
        c = cos
        return np.mat([[c, -s], [s, c]])

    def refreshVtsW(self):
        self.vtsW = self.rotMat * (self.vtsO - self.pivotO) + self.translation

    def project2Camera(self):
        self.vtsC = self.camera.project(self.vtsW)


class Arrow(Element):
    def __init__(self, l=2.0, angle=0, x=0.0, y=0.0, h=1.0, s=1.0, v=1.0, camera=None):
        Element.__init__(self, num_vts=4, x=x, y=y, camera=camera)
        self.l = l
        self.rad = np.deg2rad(angle)
        self.start = np.mat([[x], [y]])
        self.end = np.mat([[x + l * np.cos(self.rad)], y + l * np.sin(self.rad)])
        self.wx = 0.2
        self.wy = 0.1
        self.setRotMatRad(self.rad)
        self.setTranslation(x, y)
        self.init_vtsO()
        self.refreshVtsW()
        self.hsv = [h, s, v]

    def init_vtsO(self):
        self.vtsO = np.mat([[0, self.l, self.l - self.wx, self.l - self.wx], [0, 0, self.wy, -self.wy]])

    def setLength(self, l):
        self.l = l
        self.end = np.mat([[self.start[0, 0] + l * np.cos(self.rad)], self.start[1, 0] + l * np.sin(self.rad)])
        self.init_vtsO()
        self.refreshVtsW()

    def draw(self):
        if self.camera == None:
            gl.glLineWidth(lw)
            rgb = mcol.hsv_to_rgb(self.hsv)
            gl.glColor4f(rgb[0], rgb[1], rgb[2], 0.5)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(self.vtsW[0, 0], self.vtsW[1, 0])
            gl.glVertex2f(self.vtsW[0, 1], self.vtsW[1, 1])
            gl.glEnd()
            gl.glBegin(gl.GL_LINE_STRIP)
            gl.glVertex2f(self.vtsW[0, -1], self.vtsW[1, -1])
            gl.glVertex2f(self.vtsW[0, -3], self.vtsW[1, -3])
            gl.glVertex2f(self.vtsW[0, -2], self.vtsW[1, -2])
            gl.glEnd()
        else:
            self.project2Camera()
            gl.glLineWidth(lw)
            rgb = mcol.hsv_to_rgb(self.hsv)
            gl.glColor4f(rgb[0], rgb[1], rgb[2], 0.5)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(self.vtsC[0, 0], self.vtsC[1, 0])
            gl.glVertex2f(self.vtsC[0, 1], self.vtsC[1, 1])
            gl.glEnd()
            gl.glBegin(gl.GL_LINE_STRIP)
            gl.glVertex2f(self.vtsC[0, -1], self.vtsC[1, -1])
            gl.glVertex2f(self.vtsC[0, -3], self.vtsC[1, -3])
            gl.glVertex2f(self.vtsC[0, -2], self.vtsC[1, -2])
            gl.glEnd()


class Circle:
    def __init__(self, x=0.0, y=0.0, radius=1.0, h=0.0, s=1., v=1., alpha=1., camera=None):
        self.x = x
        self.y = y
        self.radius = radius

        self.num_vts = 50
        self.vts = np.zeros((2, self.num_vts))
        self.camera = camera
        self.vtsC = np.zeros((2, self.num_vts))
        self.hsv = [h, s, v]
        self.alpha = alpha

        self.theta = 2 * np.pi / self.num_vts

        self.refreshVts()

    def refreshVts(self):
        for i in range(self.num_vts):
            self.vts[0, i] = self.x + self.radius * np.cos(i * self.theta)
            self.vts[1, i] = self.y + self.radius * np.sin(i * self.theta)

    def setPosition(self, x, y):
        self.x = x
        self.y = y
        self.refreshVts()

    def setRadius(self, r):
        self.radius = r
        self.refreshVts()

    def getPosition(self):
        return [self.x, self.y]

    def setColorH(self, h=0.):
        self.hsv[0] = h

    def setColor(self, color = np.ones(4)):
        self.hsv = color[:3]
        self.alpha = color[3]

    def setColorV(self, v):
        self.hsv[2] = v

    def draw(self):
        gl.glLineWidth(lw)
        rgb = mcol.hsv_to_rgb(self.hsv)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_BLEND)
        gl.glColor4f(rgb[0], rgb[1], rgb[2], self.alpha)
        gl.glBegin(gl.GL_LINE_LOOP)
        if self.camera == None:
            for i in range(self.num_vts):
                gl.glVertex2f(self.vts[0, i], self.vts[1, i])
            gl.glEnd()
        else:
            self.vtsC = self.camera.project(self.vts)
            for i in range(self.num_vts):
                gl.glVertex2f(self.vtsC[0, i], self.vtsC[1, i])
            gl.glEnd()

class Curve:
    def __init__(self, x0=0., y0=0., x1=1., y1=1., dx0=1.5, dy0=0., dx1=1.5, dy1=0.,
                 offset0=0., offset1=0.,h=0., s=1., v=1., alpha=1., camera=None):
        self.params = [dx0 + dx1 + 2*x0 - 2*x1, 3*x1 - dx1 - 3*x0 - 2*dx0, dx0, x0,
                       dy0 + dy1 + 2*y0 - 2*y1, 3*y1 - dy1 - 3*y0 - 2*dy0, dy0, y0]
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.dx0 = dx0
        self.dy0 = dy0
        self.dx1 = dx1
        self.dy1 = dy1
        self.offset0 = offset0
        self.offset1 = offset1
        self.numt = 30
        self.startPoint = 0
        self.endPoint = self.numt
        self.t = np.linspace(0, 1, self.numt)
        self.vts = np.zeros((2, self.numt))
        self.vtsC = np.zeros((2, self.numt))
        self.hsv = [h, s, v]
        self.alpha = alpha
        self.camera = camera

        self.refreshVts()

        # Animation
        self.propagationVel = 2
        self.inputColor = [h, s, v, alpha]
        self.color = (np.ones((4, self.numt)).T*self.inputColor).T
        self.signals = np.zeros(self.numt)

    def setInputColorH(self, h):
        self.color[0, self.startPoint] = h

    def setInputColorS(self, s):
        self.color[1, self.startPoint] = s

    def setInputColorV(self, v):
        self.color[2, self.startPoint] = v


    def setInputColorAlpha(self, a):
        self.color[3, self.startPoint] = a


    def setInputColor(self, color=np.ones(4)):
        self.color[:, self.startPoint] = color

    def setPropagationVel(self, v):
        self.propagationVel = int(v)

    def setInputSignal(self, s):
        self.signals[self.startPoint] = s

    def get_outputSignal(self):
        return self.signals[self.endPoint-1]


    def refreshVts(self):
        for i in range(self.numt):
            t = self.t[i]
            tvec = np.array([t ** 3, t ** 2, t, 1.0])
            x = np.dot(self.params[0:4], tvec)
            y = np.dot(self.params[4:], tvec)
            self.vts[0, i] = x
            self.vts[1, i] = y
            if self.offset0 != 0. or self.offset1 != 0.:
                l0 = np.linalg.norm([x-self.x0, y-self.y0])
                l1 = np.linalg.norm([self.x1-x, self.y1-y])
                if self.offset0 != 0 and self.startPoint == 0 and l0 >= self.offset0:
                    self.startPoint = i
                if self.offset1 != 0 and self.endPoint == self.numt and l1 <= self.offset1:
                    self.endPoint = i

    def refreshParams(self):
        self.params = [self.dx0 + self.dx1 + 2 * self.x0 - 2 * self.x1, 3 * self.x1 - self.dx1 - 3 * self.x0 - 2 * self.dx0,
                       self.dx0, self.x0,
                       self.dy0 + self.dy1 + 2 * self.y0 - 2 * self.y1, 3 * self.y1 - self.dy1 - 3 * self.y0 - 2 * self.dy0,
                       self.dy0, self.y0]
    def setPoint0(self, x0, y0):
        self.x0 = x0
        self.y0 = y0
        self.refreshParams()
        self.refreshVts()

    def setPoint1(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        self.refreshParams()
        self.refreshVts()

    def setDPoint0(self, dx0, dy0):
        self.dx0 = dx0
        self.dy0 = dy0
        self.refreshParams()
        self.refreshVts()

    def setDPoint1(self, dx1, dy1):
        self.dx1 = dx1
        self.dy1 = dy1
        self.refreshParams()
        self.refreshVts()

    def getPoint0(self):
        return [self.x0, self.y0]

    def getPoint1(self):
        return [self.x1, self.y1]

    def signal2colorV(self, s):
        v = (2. * sigmoid(s) - 1.) * .5
        return v

    def draw(self):
        gl.glLineWidth(1)
        rgb = mcol.hsv_to_rgb(self.hsv)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_BLEND)
        gl.glColor4f(rgb[0], rgb[1], rgb[2], self.alpha)
        gl.glBegin(gl.GL_LINE_STRIP)
        if self.camera == None:
            for i in range(self.endPoint-self.startPoint):
                gl.glVertex2f(self.vts[0, self.startPoint+i], self.vts[1, self.startPoint+i])
            gl.glEnd()
        else:
            self.vtsC = self.camera.project(self.vts)
            for i in range(self.endPoint-self.startPoint):
                id = self.endPoint-i-1
                self.signals[id] = self.signals[max(self.startPoint, id-self.propagationVel)]
                self.color[:, id] = self.color[:, max(self.startPoint, id-self.propagationVel)]
                col = mcol.hsv_to_rgb([self.color[0, id], self.color[1, id], self.signals[id]])#self.color[2, id] + self.signal2colorV(self.signals[id])])
                gl.glColor4f(col[0], col[1], col[2], self.color[3, id])
                gl.glVertex2f(self.vtsC[0, id], self.vtsC[1, id])
            gl.glEnd()


class Spring(Element):
    def __init__(self, length=1.0, angle=0.0, coils=8, width=0.2, k=1.0, x=0.0, y=0.0, h=1.0, s=0.0, v=1.0,
                 camera=None):
        Element.__init__(self, num_vts=coils * 2 + 4, x=x, y=y, camera=camera)
        self.length = length  # Length of the spring
        self.coils = coils  # Number of coils of the spring
        self.width = width  # Radius of the spring
        self.k = k
        self.rad = np.deg2rad(angle)  # Direction of the spring
        self.start = np.mat([[x], [y]])  # Start point of the spring
        self.end = self.start + np.mat([[self.length * np.cos(self.rad)], [self.length * np.sin(self.rad)]])
        self.vec = self.end - self.start  # The vector represent the spring
        self.init_vtsO()
        self.setRotMatRad(self.rad)
        self.refreshVtsW()
        self.hsv = [h, s, v]  # Color of the spring

    def init_vtsO(self):
        self.vtsO[0, :] = np.linspace(0.0, self.length, self.num_vts, True)
        dl = self.length / (self.num_vts + 1)
        self.vtsO[0, 1] += 0.5 * dl
        self.vtsO[0, -2] -= 0.5 * dl
        self.vtsO[1, 2:-3:2] = 0.5 * self.width
        self.vtsO[1, 3:-2:2] = -0.5 * self.width

    def setLength(self, length):
        self.vtsO[0, :] = np.linspace(0.0, length, self.num_vts, True)
        dl = length / (self.num_vts + 1)
        self.vtsO[0, 1] += 0.5 * dl
        self.vtsO[0, -2] -= 0.5 * dl
        self.refreshVtsW()

    def resetLength(self):
        self.vtsO[0, :] = np.linspace(0.0, self.length, self.num_vts, True)
        dl = self.length / (self.num_vts + 1)
        self.vtsO[0, 1] += 0.5 * dl
        self.vtsO[0, -2] -= 0.5 * dl
        self.refreshVtsW()

    def setStart(self, x=0.0, y=0.0):
        self.start = np.mat([[x], [y]])
        self.vec = self.end - self.start
        self.length = np.linalg.norm(self.vec)
        self.rad = np.arctan2(self.vec[1, 0], self.vec[0, 0])
        self.setTranslation(x, y)
        self.setRotMatRad(self.rad)
        self.setLength(self.length)

    def setEnd(self, x=0.0, y=0.0):
        self.end = np.mat([[x], [y]])
        self.vec = self.end - self.start
        self.length = np.linalg.norm(self.vec)
        self.rad = np.arctan2(self.vec[1, 0], self.vec[0, 0])
        self.setRotMatRad(self.rad)
        self.setLength(self.length)

    def setStartEnd(self, x, y, x1, y1):
        self.start[0, 0] = x
        self.start[1, 0] = y
        self.end[0, 0] = x1
        self.end[1, 0] = y1
        self.vec = self.end - self.start
        self.length = np.linalg.norm(self.vec)
        self.rad = np.arctan2(self.vec[1, 0], self.vec[0, 0])
        self.setTranslation(x, y)
        self.setRotMatRad(self.rad)
        self.setLength(self.length)

    def draw(self):
        gl.glLineWidth(lw)
        rgb = mcol.hsv_to_rgb(self.hsv)
        gl.glColor4f(rgb[0], rgb[1], rgb[2], 0.5)
        gl.glBegin(gl.GL_LINE_STRIP)
        if self.camera == None:
            for i in range(self.num_vts):
                gl.glVertex2f(self.vtsW[0, i], self.vtsW[1, i])
            gl.glEnd()
        else:
            self.project2Camera()
            for i in range(self.num_vts):
                gl.glVertex2f(self.vtsC[0, i], self.vtsC[1, i])
            gl.glEnd()


def limit(x, lim):
    return max(min(x, lim), -lim)


# Define ground function below
def ground(x):
    return 0


class Ground(Element):
    def __init__(self, height=0.0, l=20.0, w=0.1, x=0.0, y=0.0, h=0.0, s=0.0, v=1.0, alpha=0.5, camera=None):
        self.lnum = int((l - 2 * w) / w)
        Element.__init__(self, self.lnum * 2 + 2, x=x, y=y, camera=camera)
        self.height = height
        self.length = l
        self.width = w

        self.hsv = [h, s, v]
        self.alpha = alpha
        self.initVtsO()

    def initVtsO(self):
        self.vtsO[0, 0:self.lnum] = np.linspace(0, self.length - 2 * self.width, self.lnum) + 2 * self.width
        self.vtsO[0, self.lnum:2 * self.lnum] = np.linspace(0, self.length - 2 * self.width, self.lnum) + self.width
        self.vtsO[0, -1] = self.length
        self.vtsO[1, 0:self.lnum] -= self.width
        self.vtsO[0, :] -= self.length * 0.5
        self.vtsO[1, :] += self.height
        self.refreshVtsW()

    def draw(self):
        self.project2Camera()
        gl.glLineWidth(lw)
        rgb = mcol.hsv_to_rgb(self.hsv)
        gl.glColor4f(rgb[0], rgb[1], rgb[2], self.alpha)
        gl.glBegin(gl.GL_LINES)
        for i in range(self.lnum):
            gl.glVertex2f(self.vtsC[0, i], self.vtsC[1, i])
            gl.glVertex2f(self.vtsC[0, self.lnum + i], self.vtsC[1, self.lnum + i])
        gl.glVertex2f(self.vtsC[0, -2], self.vtsC[1, -2])
        gl.glVertex2f(self.vtsC[0, -1], self.vtsC[1, -1])
        gl.glEnd()


class MassSpring:
    def __init__(self, r=0.2, l=1.2, rad=-np.pi * 0.5, x=0.0, y=3., vx=0.0, vy=0.0, dt=0.01, camera=None):
        self.r = r  # Radius of the head (m)
        self.l = l  # Length of the leg (m)
        self.rad = rad  # Direction of the leg (rad)
        self.m = 1.0  # Mass (kg)
        self.k = 1000.0  # Elastic coefficient of the spring
        self.i = 0.1  # Rotational inertia

        self.head = Circle(x, y, r, camera=camera)
        self.leg = Spring(l - r, self.rad, 6, 0.15 * self.l,
                          x=x + r * np.cos(rad), y=y + r * np.sin(rad), camera=camera)
        # Neural network
        self.nn = NeuralNet(camera=camera)
        self.input_normalized = np.zeros((6,1))
        self.nn_vx_range = 10.0
        self.nn_vy_range = 10.
        self.nn_y_min = .2
        self.nn_y_max = 5.
        self.nn_rad_min = -np.pi * 5. / 6.
        self.nn_rad_max = -np.pi / 6.

        self.vx_dst = 0.
        self.vy_dst = 0.
        self.y_dst = 2.
        self.rad_dst = -np.pi*.5
        # There are 3 states of the mass-spring
        # state 0: the mass-spring is flying
        # state 1: the mass-spring is standing
        # state 2: the mass-spring is lying down
        # state 3: hold in sky
        self.state = 0
        self.x = x  # Position x of the head
        self.y = y  # Position y of the head
        self.x0 = x + l * np.cos(rad)  # Position x of foot
        self.y0 = y + l * np.sin(rad)  # Position y of foot
        self.vec = np.mat([self.x - self.x0, self.y - self.y0]).T  # Vector that point from foot to head
        self.norm_vec = np.linalg.norm(self.vec)
        self.vx = vx  # Velocity x of the head
        self.vy = vy  # Velocity y of the head
        self.vrad = 0.0  # Angular velocity of the leg
        self.u = 0.0  # Control input of the leg direction
        self.dt = dt  # Simulation time step of the mass-spring system
        self.f = np.mat([0, 0]).T  # Force of the spring

        self.lim_vrad = 500.0 * np.pi  # Limit of angular velocity

        self.dx = 0.0  # Change of x position of the head in a time step
        self.dy = 0.0  # Change of y position of the head in a time step
        self.dvx = 0.0  # Change of velocity x of the head in a time step
        self.dvy = 0.0  # Change of velocity y of the head in a time step
        self.drad = 0.0  # Change of angle of the leg
        self.dvrad = 0.0  # Change of angular velocity of the leg

        # Measurement data
        self.control = False
        self.err = 0.0  # Angular error
        self.lerr = 0.0  # Last time step angular error
        self.derr = 0.0  # Change of angular error
        self.ierr = 0.0  # Integration of angular error
        self.errv = 0.0  # Angular error
        self.lerrv = 0.0  # Last time step angular error
        self.derrv = 0.0  # Change of angular error
        # PID controller parameters
        self.kp = 10.0  # Proportionality coefficient
        self.ki = 0.0  # Integral coefficient
        self.kd = 10.0  # Differential coefficient
        self.kpv = 2.0  # Velocity proportionality coefficient
        self.kdv = 1.0  # Velocity differential coefficient
        self.ilim = 1000.0  # Limit of integral term
        self.ulim = 1000.0  # Limit of output

        self.pe = self.m * gravity * self.y  # Potential energy
        self.ke = 0.5 * self.m * (self.vx ** 2 + self.vy ** 2)  # Kinetic energy
        self.energy = self.pe + self.ke  # Energy

    def init(self, x, y, rad, vx, vy):
        self.state = 0
        self.x = x
        self.y = max(self.l, y)
        self.rad = rad
        self.vx = vx
        self.vy = vy

    def construct_neural_network_from_file(self, name='mass_spring_neural_network.npz'):
        self.nn.construct_from_file(name)

    def network_cal_rad(self, vx=0., vy=0., y=0., vx_dst=0., vy_dst=0., y_dst=0.):
        if self.nn.dimIn == 4:
            self.input_normalized[0, 0] = remap(vx, -self.nn_vx_range, self.nn_vx_range)
            self.input_normalized[1, 0] = remap(y, -self.nn_y_min, self.nn_y_max)
            self.input_normalized[2, 0] = remap(vx_dst, -self.nn_vx_range, self.nn_vx_range)
            self.input_normalized[3, 0] = y_dst#remap(y_dst, -self.nn_y_min, self.nn_y_max)
            self.rad_dst = remap(self.nn.cal_output(self.input_normalized[:4])[0, 0], 0., 1., self.nn_rad_min,
                                 self.nn_rad_max)

        elif self.nn.dimIn == 5:
            self.input_normalized[0, 0] = remap(vx, -self.nn_vx_range, self.nn_vx_range)
            self.input_normalized[1, 0] = remap(vy, -self.nn_vy_range, self.nn_vy_range)
            self.input_normalized[2, 0] = remap(y, self.nn_y_min, self.nn_y_max)
            self.input_normalized[3, 0] = remap(vx_dst, -self.nn_vx_range, self.nn_vx_range)
            self.rad_dst = remap(self.nn.cal_output(self.input_normalized[:5])[0, 0], 0., 1., self.nn_rad_min,
                                 self.nn_rad_max)

        elif self.nn.dimIn == 6:
            self.input_normalized[0, 0] = remap(vx, -self.nn_vx_range, self.nn_vx_range)
            self.input_normalized[1, 0] = remap(vy, -self.nn_vy_range, self.nn_vy_range)
            self.input_normalized[2, 0] = remap(y, self.nn_y_min, self.nn_y_max)
            self.input_normalized[3, 0] = remap(vx_dst, -self.nn_vx_range, self.nn_vx_range)
            self.input_normalized[4, 0] = remap(vy_dst, -self.nn_vy_range, self.nn_vy_range)
            self.input_normalized[5, 0] = remap(y_dst, self.nn_y_min, self.nn_y_max)
            self.rad_dst = remap(self.nn.cal_output(self.input_normalized)[0, 0], 0., 1., self.nn_rad_min, self.nn_rad_max)

    def set_jump_dst(self, vx_dst=0., vy_dst=0., y_dst=.8):
        self.vx_dst = vx_dst
        self.vy_dst = vy_dst
        self.y_dst = y_dst

    def adjustVelocity(self):
        self.pe = self.m * gravity * self.y
        self.ke = self.energy - self.pe
        adj_vnorm = np.sqrt(2 * self.ke / self.m)
        vnorm = np.sqrt(self.vx ** 2 + self.vy ** 2)
        k = adj_vnorm / vnorm
        self.vx *= k
        self.vy *= k

    def updateState(self):
        # print(self.state)
        # When the mass-spring is flying
        if self.state == 0:
            self.dx = self.vx * self.dt
            self.dy = self.vy * self.dt
            self.dvx = 0
            self.dvy = -gravity * self.dt
            self.drad = self.vrad * self.dt
            self.dvrad = self.u / self.i * self.dt

            self.x += self.dx
            self.y += self.dy
            self.vx += self.dvx
            self.vy += self.dvy
            self.rad += self.drad
            self.vrad += self.dvrad
            if self.rad > np.pi:
                self.rad -= 2.0 * np.pi
            if self.rad < - np.pi:
                self.rad += 2.0 * np.pi
            self.vrad += self.dvrad
            self.vrad = limit(self.vrad, self.lim_vrad)
            self.x0 = self.x + self.l * np.cos(self.rad)
            self.y0 = self.y + self.l * np.sin(self.rad)
            self.vec = np.mat([self.x - self.x0, self.y - self.y0]).T

            # Condition of changing state
            # flying -> stop
            if ground(self.x) >= self.y - self.r:
                self.state = 2
                self.y = self.r

            # flying -> standing
            if ground(self.x0) >= self.y0:
                self.state = 1
                self.y0 = 0
                self.norm_vec = np.linalg.norm(self.vec)
                self.f = self.vec * self.k * (self.l / self.norm_vec - 1.0)
                self.adjustVelocity()
        # When the mass-spring is standing
        if self.state == 1:
            self.dx = self.vx * self.dt
            self.dy = self.vy * self.dt
            self.dvx = self.f[0, 0] / self.m * self.dt
            self.dvy = (self.f[1, 0] / self.m - gravity) * self.dt

            self.x += self.dx
            self.y += self.dy
            self.vx += self.dvx
            self.vy += self.dvy
            self.vec = np.mat([self.x - self.x0, self.y - self.y0]).T
            self.norm_vec = np.linalg.norm(self.vec)
            self.rad = np.arctan2(-self.vec[1, 0], -self.vec[0, 0])
            self.f = self.vec * self.k * (self.l / self.norm_vec - 1.0)

            # Condition of changing state
            # standing -> flying
            if self.norm_vec >= self.l:
                self.state = 0
                self.y0 = 0
                self.adjustVelocity()
                if self.control:
                    if self.nn.dimIn == 4:
                        delta_y = .5*self.vy**2/gravity
                        y_top = delta_y+self.y
                        self.network_cal_rad(vx=self.vx, y=y_top,
                                             vx_dst=self.vx_dst, y_dst=self.y_dst)
                    elif self.nn.dimIn == 6:
                        self.network_cal_rad(vx=self.vx, vy=self.vy, y=self.y,
                                             vx_dst=self.vx_dst, vy_dst=self.vy_dst, y_dst=self.y_dst)
                # print('vx: '+str(round(self.vx, 3)))

            # standing -> stop
            if ground(self.x) >= self.y - self.r:
                self.state = 2
                self.y = self.r

        if self.state == 3:
            self.drad = self.vrad * self.dt
            self.dvrad = self.u / self.i * self.dt
            self.rad += self.drad
            self.vrad += self.dvrad
            self.x0 = self.x + self.l * np.cos(self.rad)
            self.y0 = self.y + self.l * np.sin(self.rad)

    def hold(self):
        self.state = 3

    def updateInput(self, dstRad):
        if not self.control:
            self.u = 0
        else:
            self.err = dstRad - self.rad
            while self.err > np.pi:
                self.err -= np.pi * 2.0
            while self.err < -np.pi:
                self.err += np.pi * 2.0
            self.ierr += self.err
            self.ierr = limit(self.ierr, self.ilim)
            self.derr = self.err - self.lerr
            self.lerr = self.err

            dstv = self.kp * self.err + self.ki * self.ierr + self.kd * self.derr
            self.errv = dstv - self.vrad
            self.derrv = self.errv - self.lerrv
            self.lerrv = self.errv
            self.u = self.kpv * self.errv + self.kdv * self.derrv
            self.u = limit(self.u, self.ulim)

    def update(self, dt=0.01, control=True):
        self.control = control
        self.dt = dt
        if control:
            self.updateInput(self.rad_dst)
        self.updateState()

    def draw(self):
        self.head.setPosition(self.x, self.y)
        self.leg.setStartEnd(self.x + self.r * np.cos(self.rad), self.y + self.r * np.sin(self.rad),
                             self.x0, self.y0)
        # self.leg.setStart(self.x, self.y)
        # self.leg.setEnd(self.x0, self.y0)
        self.head.draw()
        self.leg.draw()


# Neural Network
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def dsigmoid(s):
    return s * (1 - s)


def relu(a):
    k = .3
    kk = 0.01 * k
    a[a >= 0] *= k
    a[a < 0] *= kk
    return a


def drelu(r):
    k = .3
    kk = 0.01 * k
    r[r >= 0] = k
    r[r < 0] = kk
    return r

def map2color(x, h0=.59, h1=.03):
    k = 10.
    s1 = sigmoid(k*x)
    s0 = 1. - s1
    rad0 = h0*twoPi
    rad1 = h1*twoPi
    vec0 = np.array([np.cos(rad0), np.sin(rad0)])
    vec1 = np.array([np.cos(rad1), np.sin(rad1)])
    vec = s0*vec0 + s1*vec1
    h = (np.arctan2(vec[1], vec[0]))/twoPi
    if h < 0:
        h += 1
    s = np.linalg.norm(vec)
    return [h, s, .75*s, 1.]


twoPi = 2.*np.pi
#  z = w*x + b
#  y = sigmoid(z)
class Layer:
    def __init__(self, dimX=3, dimY=2, actFunc=1):
        self.dimX = dimX
        self.dimY = dimY
        self.weightRange = 6.
        self.biasRange = 3.
        self.weight = self.weightRange * (np.random.random((self.dimY, self.dimX)) - 0.5)
        self.bias = self.biasRange*(np.random.random((self.dimY, 1)) - 0.5)
        self.actFunc = actFunc  # Activation function:  1:sigmoid   2:relu

        self.x = np.zeros((self.dimX, 1))
        self.y = np.zeros((self.dimY, 1))
        self.djz = np.zeros((self.dimY, 1))
        self.djw = np.zeros((self.dimY, self.dimX))
        self.djb = np.zeros((self.dimY, 1))
        self.djy = np.zeros((self.dimY, 1))
        self.djx = np.zeros((self.dimX, 1))

        # Graph
        self.nodes = np.zeros(self.dimY, dtype=Circle)
        self.edges = np.zeros((self.dimY, self.dimX), dtype=Curve)
        # Color map
        h0 = 0.59
        h1 = 0.03
        self.k = 1.
        rad0 = h0*twoPi
        rad1 = h1*twoPi
        self.vec0 = np.array([np.cos(rad0), np.sin(rad0)])
        self.vec1 = np.array([np.cos(rad1), np.sin(rad1)])

        self.nodesColor = np.ones((self.dimY, 4))

    def random_init_parameters(self):
        self.weight = self.weightRange * (np.random.random((self.dimY, self.dimX)) - 0.5)
        self.bias = self.biasRange * (np.random.random((self.dimY, 1)) - 0.5)

    def get_parameter(self):
        return [self.weight, self.bias, self.actFunc]

    def set_parameter(self, parameter):
        self.weight = parameter[0]
        self.bias = parameter[1]
        self.actFunc = parameter[2]

    def map2nodesColor(self, v):
        s1 = v#sigmoid(self.k * v)
        s0 = 1 - s1
        vecs = s0*self.vec0 + s1*self.vec1
        h = np.arctan2(vecs[:, 1], vecs[:, 0])/twoPi
        h[h < 0.] += 1.
        s = np.linalg.norm(vecs, axis=1)
        self.nodesColor[:, 0] = h
        self.nodesColor[:, 1] = s
        self.nodesColor[:, 2] = s
        # self.nodesColor[:, 3] = 1.

    def setActFunc(self, actFunc):
        self.actFunc = actFunc

    def setActFuncToSigmoid(self):
        self.actFunc = 1

    def setActFuncToRelu(self):
        self.actFunc = 2

    def setDimX(self, dim):
        self.dimX = dim
        self.weight = np.random.random((self.dimY, self.dimX))

    def setDimY(self, dim):
        self.dimY = dim
        self.weight = np.random.random((self.dimY, self.dimX))
        self.bias = np.random.random((self.dimY, 1))

    def cal_output(self, x):
        self.x = x
        if self.actFunc == 1:  # Sigmoid activation function
            self.y = self.cal_outputSigmoid(self.x)
        elif self.actFunc == 2:  # Relu actiation function
            self.y = self.cal_outputRelu(self.x)
        return self.y

    def cal_outputSigmoid(self, x):
        z = np.dot(self.weight, x) + self.bias
        return sigmoid(z)

    def cal_outputRelu(self, x):
        return relu(np.dot(self.weight, x) + self.bias)

    def backpropagation(self, djy, alpha):
        n = self.djy.shape[1]
        self.djy = djy
        if self.actFunc == 1:  # Sigmoid activation function
            self.djz = self.djy * dsigmoid(self.y)
        elif self.actFunc == 2:  # Relu activation function
            self.djz = self.djy * drelu(self.y)
        self.djw = np.dot(self.djz, self.x.T) / n
        self.djb = (np.sum(self.djz, 1) / n).reshape((self.dimY, 1))
        self.djx = np.dot(self.weight.T, self.djz)

        self.weight -= alpha * self.djw
        self.bias -= alpha * self.djb
        return self.djx

    def set_edge_input(self, x):
        for i in range(self.dimY):
            for j in range(self.dimX):
                self.edges[i, j].setInputSignal(x[j])

    def get_output(self):
        return self.y

    def init_graph(self, x=0., y=0., r=.2, camera=None):
        b = 6. * r
        d = 3. * r
        offsetX = .5 * b
        offsetYNode = .5*(self.dimY-1)*d
        offsetYedge = .5*(self.dimX-1)*d
        self.map2nodesColor(self.bias)
        for i in range(self.dimY):
            self.nodes[i] = Circle(x=x+offsetX, y=y+d*i-offsetYNode, radius=r,
                                   h=self.nodesColor[i, 0], s=self.nodesColor[i, 1], v=self.nodesColor[i, 2],
                                   alpha=self.nodesColor[i, 3], camera=camera)
            for j in range(self.dimX):
                colorEdge = map2color(self.weight[i, j])
                self.edges[i, j] = Curve(x0=x-b+offsetX, y0=y+j*d-offsetYedge, x1=x+offsetX, y1=y+d*i-offsetYNode,
                                         offset0=r, offset1=r,
                                         h=colorEdge[0], s=colorEdge[1], v=colorEdge[2],
                                         alpha=colorEdge[3], camera=camera)
    def draw(self):
        x = np.zeros((self.dimX, 1))
        for j in range(self.dimX):
            x[j, 0] = self.edges[0, j].get_outputSignal()
        y = self.cal_output(x)
        self.map2nodesColor(y)
        for i in range(self.dimY):
            self.nodes[i].setColor(self.nodesColor[i, :])
            self.nodes[i].draw()
            for j in range(self.dimX):
                self.edges[i, j].draw()


# Neural networks class
# dimIn: dimension of input
# dimOut: dimension of output
# ls: number of neures in every hidden layer
class NeuralNet:
    def __init__(self, dimIn=2, dimOut=3, ls=np.array([3, 3, 3]), camera=None):
        self.ls = ls
        self.numLs = len(ls) + 1  # Layers count
        self.layers = []
        self.dimIn = dimIn
        self.dimOut = dimOut
        dimx = dimIn
        for i in range(self.numLs - 1):
            dimy = ls[i]
            l = Layer(dimx, dimy, actFunc=2)
            dimx = dimy
            self.layers.append(l)
        l = Layer(dimx, dimOut, actFunc=2)
        self.layers.append(l)

        # Training information
        self.yest = []  # Estimation of output
        self.err = []  # Error between estimation and output data
        self.j = 0  # Cost function output
        self.js = []  # All cost in training loop

        # Graph
        self.camera = camera
        self.inputNode = np.zeros(self.dimIn, dtype=Circle)
        self.edgeInput = np.zeros(self.dimIn)
        self.init_graph()

    def random_init_parameters(self):
        for l in self.layers:
            l.random_init_parameters()

    def save_network_structure(self, name='neural_network'):
        ls = self.ls
        num_ls = self.numLs
        dim_in = self.dimIn
        dim_out = self.dimOut
        layers_parameters = []
        for l in self.layers:
            layers_parameters.append(l.get_parameter())
        np.savez(name, ls=ls, num_ls=num_ls, dim_in=dim_in, dim_out=dim_out,
                 layers_parameters=layers_parameters)

    def construct_from_file(self, name='neural_network.npz'):
        params = np.load(name, allow_pickle=True)
        self.ls = params['ls']
        self.numLs = len(self.ls)+1
        self.layers = []
        self.dimIn = params['dim_in']
        self.dimOut = params['dim_out']
        layers_parameters = params['layers_parameters']


        dimx = self.dimIn
        for i in range(self.numLs -1):
            dimy = self.ls[i]
            l = Layer(dimx, dimy)
            l.set_parameter(layers_parameters[i])
            dimx = dimy
            self.layers.append(l)
        l = Layer(dimx, self.dimOut)
        l.set_parameter(layers_parameters[-1])
        self.layers.append(l)
        print('Neural Network:')
        print('dim in: '+str(self.dimIn) + '\t dim out: ' + str(self.dimOut))
        print('layers: ' + str(self.ls))

        self.init_graph()


    def init_graph(self, x=0., y=0., r=.2):
        b = 6. * r
        d = 3. * r
        offsetX = .5 * (self.numLs-1) * b
        offsetInputNodeY = .5*(self.dimIn-1) * d


        self.inputNode = np.zeros(self.dimIn, dtype=Circle)
        for i in range(self.dimIn):
            self.inputNode[i] = Circle(x=x-.5*b-offsetX, y=y+i*d-offsetInputNodeY, radius=r,
                                       s=0., v=.5, camera=self.camera)
        for i in range(self.numLs):
            self.layers[i].init_graph(x=x+b*i-offsetX, y=y, r=r, camera=self.camera)
        self.edgeInput = np.zeros(self.dimIn)


    def cal_output(self, input):
        x = input
        for i in range(self.numLs):
            x = self.layers[i].cal_output(x)
            # print('weight '+str(i))
            # print(self.layers[i].weight)
            # print('out:'+str(np.shape(x)))
            # print(x)
        return x
        # for l in self.layers:
        #     x = l.cal_output(x)
        # return x

    def cal_output_grid(self, x):
        s = x.shape
        xv = x.reshape((-1, 2)).T
        y = self.cal_output(xv)
        y = y.reshape((s[0], s[1]))
        return y

    def backpropagation(self, djy, alpha):
        djx = djy
        for i in range(self.numLs):
            djx = self.layers[-i - 1].backpropagation(djx, alpha)

    def training(self, xdata, ydata, iteration=500, alpha=0.1, detect_vg=True, dynamic_step=False):
        print('training...')
        alpha_max = .6
        alpha_min = 0.001
        alpha_min0 = alpha_min
        alpha_max0 = alpha_max
        lock_max = False
        nvb_num = 10
        vibrate = 0
        notvibrate = 0
        alpha = alpha
        alpha0 = alpha
        data_num = xdata.shape[1]
        i = 0
        t0 = time.time()
        print('t:'+str(t0))
        while i < iteration:
            t_i = time.time()-t0
            left_num = iteration-i
            left_time = int(t_i*float(left_num)/float(i+1))
            left_sec = left_time%60
            left_min = int(left_time/60)
            left_hos = int(left_min/60)
            left_min = left_min%60
            # print(xdata)
            self.yest = self.cal_output(xdata)
            self.err = self.yest - ydata
            self.j = np.sum(.5 * self.err ** 2.)
            jj = self.j/data_num
            self.js.append(self.j)
            self.backpropagation(self.err, alpha)
            j_last = 0
            if i > 5:
                j_last = self.j - (self.js[i-4]-self.j)/4*left_num
            print('iter: ' + str(i) + '/'+str(iteration)+'\tcost: '\
                  +str(round(float(self.j), 5)) + '\tcost/n: '+str(round(float(jj),8))\
                  +'\talpha: '+str(round(alpha,5))\
                  +'\tremain: '+str(left_hos)+'h'+str(left_min)+'m'+str(left_sec)+'s')
            print('\tcost end: '+str(round(j_last,3))+'\tcost/n end: '+str(round(j_last/data_num, 8))\
                  +'\talpha max: '+str(round(alpha_max,5)) + '\talpha min: '+str(round(alpha_min,5)))
            # Detect vanishing gradient
            if detect_vg and self.j > 50 and i > 5 and abs(self.js[i-4] - self.js[i-1]) < 0.0001 and alpha == alpha_max or self.j > 8000:# or self.j != self.j:
                self.js = []
                self.random_init_parameters()
                i = 0
                alpha = alpha0
                vibrate = False
                alpha_max = alpha_max0
                alpha_min = alpha_min0
                print(self.js)
                print('restart...')
            # Dynamic change step size
            if dynamic_step and i > 15:
                # print('alpha max: '+str(round(alpha_max,5)) + '\talpha min: '+str(round(alpha_min,5)))
                if self.js[i - 2] < self.js[i - 1]:
                    vibrate += 1
                    notvibrate = 0

                else:
                    notvibrate += 1
                    if notvibrate >= nvb_num:
                        lock_max = False
                        notvibrate = nvb_num
                        vibrate = 0
                if notvibrate == nvb_num:
                    if self.j > 1000 and self.js[i-2] - self.js[i-1] < 10.:
                        alpha += 0.01
                    if self.j <= 1000 and self.js[i-2] - self.js[i-1] < .1:
                        alpha += 0.001
                    alpha = min(alpha, alpha_max)
                elif self.j < 100 and self.js[i-4] - self.js[i-1] > 50. or vibrate:
                    if not lock_max:
                        alpha_max = alpha
                        lock_max = True
                        alpha_min = alpha_max * .5
                    alpha -= 0.004
                    alpha = max(alpha_min, alpha)


            i += 1

    def setInput(self, x):
        self.edgeInput = x

    def draw(self):
        for i in range(self.dimIn):
            self.inputNode[i].setColorV(self.edgeInput[i])
            self.inputNode[i].draw()
        x = self.edgeInput
        for l in range(self.numLs):
            self.layers[l].set_edge_input(x)
            self.layers[l].draw()
            x = self.layers[l].get_output()


# TEST
# Neural network test
def test_neural():
    camera = Camera(scale=100)
    camera.setY(0.0)
    nn = NeuralNet(dimIn=2, dimOut=2, ls=np.array([2, 2, 5]), camera=camera)
    nn.construct_from_file('mass_spring_neural_network.npz')
    # nn.save_network_structure()
    t = [0]
    @window.event
    def on_draw():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()

        nn.draw()

    def update(dt, t):
        t[0] += dt
        w = 1.
        s = .5*np.sin(2.*np.pi*w*t[0])+.5
        c = .5*np.cos(2.*np.pi*w*t[0])+.5
        nn.setInput(np.array([s, c, s, c, s, c]).reshape(-1, 1))
        print(nn.cal_output(np.array([s, c, s, c, s, c]).reshape(-1, 1)))
    pyglet.clock.schedule_interval(update, t=t, interval=1.0 / 100.0)
    pyglet.app.run()
# test_neural()

# Sampling Learning data
# learning data: shape:(6,num)
# [vx_last, vy_last, y_last, vx_this, vy_this, y_this, angle_hit]'
def mass_spring_sampling_data(num=10, fast_mode = False):
    def randomInit(vx_mean=0., vx_range=5., vy_mean=0., vy_range=5.,
                   y_mean=3., y_range=2., rad_mean=-np.pi*.5, rad_range=-np.pi/3.):
        y = y_mean + 2. * (np.random.random() - .5) * y_range
        vx = vx_mean + 2. * (np.random.random() - .5) * vx_range
        vy = vy_mean + 2. * (np.random.random() - .5) * vy_range
        rad = rad_mean + 2. * (np.random.random() - .5) * rad_range
        return vx,vy,y,rad

    camera = Camera(scale=90)
    camera.setY(1.5)
    jumper = MassSpring(camera=camera)
    g = Ground(l=100,camera=camera)
    num = num
    vx,vy,y,rad = randomInit()
    jumper.init(0, y, rad, vx, vy)
    sample = [vx, vy, y, 0, 0, 0, rad]
    data = np.zeros((7, num))
    state = [0, 0]          # state[0]: last state; state[1] = this state
    i = [0]

    @window.event
    def on_draw():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()
        jumper.draw()
        g.draw()

    def update(dt, sample, data, state, i, num):
        jumper.update(dt=0.01, control=False)
        # camera.setX(jumper.x)
        state[0] = state[1]
        state[1] = jumper.state
        if state[0] == 1 and state[1] != state[0] and i[0] < num:
            sample[3:-1] = [jumper.vx, jumper.vy, jumper.y]
            data[:, i[0]] = sample
            sample_display = [round(v, 3) for v in sample]
            print('sample '+str(i[0]+1)+'\tvx0,vy0,y0,vx1,vy1,y1,rad: '+str(sample_display))
            i[0] += 1
            if i[0] == num:
                name = 'mass_spring_learning_data_'+str(int(num/1000))+'_k'
                np.save(name, data)
                print('sample ok')
            vx, vy, y, rad = randomInit()
            jumper.init(0, y, rad, vx, vy)
            sample[:] = [vx, vy, y, 0, 0, 0, rad]
            state[:] = [0,0]
        elif state[1] == 2:
            vx, vy, y, rad = randomInit()
            jumper.init(0, y, rad, vx, vy)
            sample[:] = [vx, vy, y, 0, 0, 0, rad]
            state[:] = [0, 0]

    if fast_mode:
        while(i[0] <= num):
            update(dt=0.01, sample=sample, data=data, state=state, i=i, num=num)


    else:
        pyglet.clock.schedule_interval(update,interval=1./100.,
                                       sample=sample, data=data, state=state, i=i, num=num)
        pyglet.app.run()
# mass_spring_sampling_data(50000, fast_mode=True)

# Mass spring 5 input neural network
def mass_spring_learning_5_input():
    def normalize_data(input_data, output_data, vx_range=10., vy_range=10., y_min=.2, y_max=5.,
                       rad_min=-np.pi*5./6., rad_max=-np.pi/6.):
        input_data[0, :] = remap(input_data[0, :], -vx_range, vx_range)
        input_data[1, :] = remap(input_data[1, :], -vy_range, vy_range)
        input_data[2, :] = remap(input_data[2, :], y_min, y_max)
        input_data[3, :] = remap(input_data[3, :], -vx_range, vx_range)
        output_data[:] = remap(output_data[:], rad_min, rad_max)
    data = np.load('mass_spring_learning_data_100k.npy')
    data = np.array(data)
    # Save to excel
    # data_pd = pd.DataFrame(data.T, columns=['vx0', 'vy0', 'y0', 'vx1', 'vy1', 'y1', 'rad'])
    # data_pd.to_excel('mass_spring_learning_data_100k.xlsx')
    vy1 = data[4, :]
    vy1[vy1 > 3] = 2.
    vy1[(vy1 <= 3.) * (vy1 >= 0.5)] = 1.
    vy1[vy1 < 0.5] = 0.
    input_data = np.vstack([data[:4, :], vy1])
    output_data = data[-1, :]

    learning_num = 40000
    input_data = input_data[:, :learning_num]
    output_data = output_data[:learning_num]
    normalize_data(input_data, output_data)

    nn = NeuralNet(dimIn=5, dimOut=1, ls=15*np.ones(1, dtype=int))
    # nn.save_network_structure('mass_spring_nn_5_inputs_20_layers_30k_samples_1st_loop')
    nn.construct_from_file('mass_spring_nn_5_inputs_20_layers_10k_samples.npz')
    nn.training(input_data, output_data, iteration=8000, alpha=0.1, detect_vg=True)
    nn.save_network_structure(name='mass_spring_nn_5_inputs_20_layers_10k_samples')
# mass_spring_learning_5_input()

# Mass spring 4 input neural network
# input: [vx0, y0, vx1, y1];    output:[rad]
def mass_spring_learning_4_input():
    def normalize_data(input_data, output_data, vx_range=10., vy_range=10., y_min=.2, y_max=5.,
                       rad_min=-np.pi * 5. / 6., rad_max=-np.pi / 6.):
        input_data[0, :] = remap(input_data[0, :], -vx_range, vx_range)
        input_data[1, :] = remap(input_data[1, :], y_min, y_max)
        input_data[2, :] = remap(input_data[2, :], -vx_range, vx_range)
        input_data[3, :] = remap(input_data[3, :], y_min, y_max)
        output_data[:] = remap(output_data[:], rad_min, rad_max)

    data = np.load('mass_spring_learning_data_50k.npy')
    data = np.array(data)
    vx0 = data[0, :]
    vy0 = data[1, :]
    y0 = data[2, :]
    vx1 = data[3, :]
    vy1 = data[4, :]
    y1 = data[5, :]
    y0_top = y0 + .5*vy0**2/gravity
    y1[vy1 > 0] += .5*vy1[vy1>0]**2/gravity
    y1[vy1 <= 0] = 0.2
    input_data = np.vstack([vx0, y0_top, vx1, y1])
    output_data = data[-1, :]

    learning_num = 50000
    input_data = input_data[:, :learning_num]
    output_data = output_data[:learning_num]
    normalize_data(input_data, output_data)

    nn = NeuralNet(dimIn=4, dimOut=1, ls=10*np.ones(5, dtype=int))
    # nn.save_network_structure('mass_spring_nn_4_input_1st_loop')
    # nn.construct_from_file('mass_spring_nn_4_input.npz')
    nn.training(input_data, output_data, iteration=30000, alpha=0.1, detect_vg=True)
    nn.save_network_structure(name='mass_spring_nn_4_input_1')
# mass_spring_learning_4_input()

# Mass spring 4 input neural network
# input: [vx0, y0, vx1, y1_fuzzy]   output: [rad]
def mass_spring_learning_4_input_fuzzy():
    def normalize_data(input_data, output_data, vx_range=10., vy_range=10., y_min=.2, y_max=5.,
                       rad_min=-np.pi * 5. / 6., rad_max=-np.pi / 6.):
        input_data[0, :] = remap(input_data[0, :], -vx_range, vx_range)
        input_data[1, :] = remap(input_data[1, :], y_min, y_max)
        input_data[2, :] = remap(input_data[2, :], -vx_range, vx_range)
        output_data[:] = remap(output_data[:], rad_min, rad_max)

    data = np.load('mass_spring_learning_data_50k.npy')
    data = np.array(data)
    vx0 = data[0, :]
    vy0 = data[1, :]
    y0 = data[2, :]
    vx1 = data[3, :]
    vy1 = data[4, :]
    y1 = data[5, :]
    y0_top = y0 + .5*vy0**2/gravity
    y1[vy1 > 0] = 1.
    y1[vy1 <= 0] = 0.
    input_data = np.vstack([vx0, y0_top, vx1, y1])
    output_data = data[-1, :]

    learning_num = 50000
    input_data = input_data[:, :learning_num]
    output_data = output_data[:learning_num]
    normalize_data(input_data, output_data)

    nn = NeuralNet(dimIn=4, dimOut=1, ls=10*np.ones(20, dtype=int))
    # nn.save_network_structure('mass_spring_nn_4_input_1st_loop')
    nn.construct_from_file('mass_spring_nn_4_input_fuzzy_1.npz')
    nn.training(input_data, output_data, iteration=10000, alpha=0.0001, detect_vg=True, dynamic_step=True)
    nn.save_network_structure(name='mass_spring_nn_4_input_fuzzy_2')
mass_spring_learning_4_input_fuzzy()

# Mass spring Learning from learning data
def mass_spring_learning_6_input():
    def normalize_data(input_data,output_data,vx_range=10., vy_range=10., y_min=.2, y_max=5., 
                       rad_min=-np.pi*5./6., rad_max=-np.pi/6.):
        input_data[0, :] = remap(input_data[0, :], -vx_range, vx_range)
        input_data[1, :] = remap(input_data[1, :], -vy_range, vy_range)
        input_data[2, :] = remap(input_data[2, :], y_min, y_max)
        input_data[3, :] = remap(input_data[3, :], -vx_range, vx_range)
        input_data[4, :] = remap(input_data[4, :], -vy_range, vy_range)
        input_data[5, :] = remap(input_data[5, :], y_min, y_max)
        output_data[:] = remap(output_data[:], rad_min, rad_max)

    data = np.load('mass_spring_learning_data_100k.npy')
    data = np.array(data)
    # d = pd.DataFrame(data.T, columns=['vx0', 'vy0', 'y0', 'vx1', 'vy1', 'y1', 'rad'])
    # d.to_excel('mass_spring_learning_data.xls')
    learning_num = 20000
    input_data = data[:-1, :learning_num]
    output_data = data[-1, :learning_num]
    normalize_data(input_data, output_data)

    nn = NeuralNet(dimIn=6, dimOut=1, ls=10*np.ones(3, dtype=int))
    nn.construct_from_file('mass_spring_nn_5.npz')
    nn.training(input_data, output_data, iteration=4000, alpha=0.1, detect_vg=True)
    nn.save_network_structure(name='mass_spring_nn_5')
# nn = mass_spring_learning_6_input()

# Mass spring let's jump
def mass_spring_jump():
    camera = Camera(scale=80)
    camera.setY(1.5)
    ground = Ground(l=100, camera=camera)

    jumpers = []
    num = 2
    for i in range(num):
        j = MassSpring(y=3.5, camera=camera)
        j.set_jump_dst(vx_dst=2., y_dst=1.)
        j.head.setColorH(i/float(num))
        jumpers.append(j)
    jumpers[0].construct_neural_network_from_file('mass_spring_nn_4_input_fuzzy_deep_3.npz')
    jumpers[1].construct_neural_network_from_file('mass_spring_nn_4_input_fuzzy_deep_2.npz')

    t = [0]
    @window.event
    def on_draw():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()

        ground.draw()
        for j in jumpers:
            j.draw()
    def update(dt, t):
        x = 0
        for j in range(len(jumpers)):
            if jumpers[j].x < 500:
                jumpers[j].set_jump_dst(vx_dst=2., y_dst=1.)
            else:
                jumpers[j].set_jump_dst(vx_dst=0., y_dst=1.)
            jumpers[j].update(dt=dt, control=True)
            print('jumper '+str(j)+'\tx: '+str(round(jumpers[j].x,3))+'\tvx: '+str(round(jumpers[j].vx,3)))
            x += jumpers[j].x
        x /= len(jumpers)
        camera.setX(jumpers[1].x)

    pyglet.clock.schedule_interval(update, t=t, interval=1/100)
    pyglet.app.run()
# mass_spring_jump()





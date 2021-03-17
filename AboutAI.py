import pyglet.gl as gl
import numpy as np
import numpy.matlib
import matplotlib.colors as mcol
import pyglet

# O: Object   e.g. vtsO: vertexes in Object Coordinate System
# W: World    e.g. vtsW: vertexes in World Coordinate System
gravity = 9.81
w = 1080  # Screen width
h = 720  # Screen height
lw = 2.0  # Line width

config = gl.Config(sample_buffers=1, samples=8)  # Anti-aliasing
window = pyglet.window.Window(w, h, config=config)

def rad2rm(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.mat([[c, -s], [s, c]])


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

    def getOutputSignal(self):
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
    def __init__(self, r=0.2, l=1.2, rad=-np.pi * 0.5, x=0.0, y=3, vx=0.0, vy=0.0, dt=0.01, camera=None):
        self.r = r  # Radius of the head (m)
        self.l = l  # Length of the leg (m)
        self.rad = rad  # Direction of the leg (rad)
        self.m = 1.0  # Mass (kg)
        self.k = 1000.0  # Elastic coefficient of the spring
        self.i = 0.1  # Rotational inertia

        self.head = Circle(x, y, r, camera=camera)
        self.leg = Spring(l - r, self.rad, 6, 0.15 * self.l,
                          x=x + r * np.cos(rad), y=y + r * np.sin(rad), camera=camera)

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

    def updateInput(self, dstRad, control=True):
        if not control:
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

    def update(self, dstRad=None, dt=0.01, control=True):
        self.dt = dt
        self.updateInput(dstRad, control)
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
    k = 1.0
    kk = 0.01 * k
    a[a >= 0] *= k
    a[a < 0] *= kk
    return a


def drelu(r):
    k = 1.0
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
        self.map2nodesColor(self.bias)

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

    def calOutput(self, x):
        self.x = x
        if self.actFunc == 1:  # Sigmoid activation function
            self.y = self.calOutputSigmoid(self.x)
        elif self.actFunc == 2:  # Relu actiation function
            self.y = self.calOutputRelu(self.x)
        return self.y

    def calOutputSigmoid(self, x):
        z = np.dot(self.weight, x) + self.bias
        return sigmoid(z)

    def calOutputRelu(self, x):
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

    def setEdgeInput(self, x):
        for i in range(self.dimY):
            for j in range(self.dimX):
                self.edges[i, j].setInputSignal(x[j])

    def getOutput(self):
        return self.y

    def initGraph(self, x=0., y=0., r=.2, camera=None):
        b = 6. * r
        d = 3. * r
        offsetX = .5 * b
        offsetYNode = .5*(self.dimY-1)*d
        offsetYedge = .5*(self.dimX-1)*d
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
            x[j, 0] = self.edges[0, j].getOutputSignal()
        y = self.calOutput(x)
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
        self.numLs = len(ls) + 1  # Layers count
        self.layers = []
        self.dimIn = dimIn
        self.dimOut = dimOut
        dimx = dimIn
        for i in range(self.numLs - 1):
            dimy = ls[i]
            l = Layer(dimx, dimy, actFunc=1)
            dimx = dimy
            self.layers.append(l)
        l = Layer(dimx, dimOut, actFunc=1)
        self.layers.append(l)

        # Training information
        self.yest = []  # Estimation of output
        self.err = []  # Error between estimation and output data
        self.j = 0  # Cost function output
        self.js = []  # All cost in training loop

        # Graph
        self.camera = camera
        self.inputNode = np.zeros(self.dimIn, dtype=Circle)
        self.initGraph()
        self.edgeInput = np.zeros(self.dimIn)

    def initGraph(self, x=0., y=0., r=.2):
        b = 6. * r
        d = 3. * r
        offsetX = .5 * (self.numLs-1) * b
        offsetInputNodeY = .5*(self.dimIn-1) * d
        for i in range(self.dimIn):
            self.inputNode[i] = Circle(x=x-.5*b-offsetX, y=y+i*d-offsetInputNodeY, radius=r,
                                       s=0., v=.5, camera=self.camera)
        for i in range(self.numLs):
            self.layers[i].initGraph(x=x+b*i-offsetX, y=y, r=r, camera=self.camera)

    def calOutput(self, input):
        x = input
        for i in range(self.numLs):
            x = self.layers[i].calOutput(x)
        return x

    def calOutputGrid(self, x):
        s = x.shape
        xv = x.reshape((-1, 2)).T
        y = self.calOutput(xv)
        y = y.reshape((s[0], s[1]))
        return y

    def backpropagation(self, djy, alpha):
        djx = djy
        for i in range(self.numLs):
            djx = self.layers[-i - 1].backpropagation(djx, alpha)

    def training(self, xdata, ydata, iteration=500, alpha=0.1):
        for i in range(iteration):
            self.yest = self.calOutput(xdata)
            self.err = self.yest - ydata
            self.j = np.sum(0.5 * self.err ** 2)
            self.js.append(self.j)
            self.backpropagation(self.err, alpha)

    def setInput(self, x):
        self.edgeInput = x

    def draw(self):
        for i in range(self.dimIn):
            self.inputNode[i].setColorV(self.edgeInput[i])
            self.inputNode[i].draw()
        x = self.edgeInput
        for l in range(self.numLs):
            self.layers[l].setEdgeInput(x)
            self.layers[l].draw()
            x = self.layers[l].getOutput()


# TEST
# Neural network test
def test_neural():
    camera = Camera(scale=130)
    camera.setY(0.0)
    nn = NeuralNet(dimIn=2, dimOut=2, ls=np.array([7, 6, 5]), camera=camera)
    t = [0]
    @window.event
    def on_draw():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()

        nn.draw()

    def update(dt):
        global t
        t += dt
        w = 1.
        s = .5*np.sin(2.*np.pi*w*t)+.5
        c = .5*np.cos(2.*np.pi*w*t)+.5
        nn.setInput(np.array([s, c]))
    pyglet.clock.schedule_interval(update, 1.0 / 100.0)
    pyglet.app.run()

# test_neural()


# Sampling Learning data
# learning data: shape:(6,num)
# [vx_last, vy_last, y_last, vx_this, vy_this, y_this, angle_hit]'
def mass_spring_sampling_data(num=10):
    def randomInit(vx_mean=0., vx_range=3., vy_mean=0., vy_range=2.,
                   y_mean=2., y_range=1., rad_mean=-np.pi*.5, rad_range=-np.pi/3.):
        y = y_mean + 2. * (np.random.random() - .5) * y_range
        vx = vx_mean + 2. * (np.random.random() - .5) * vx_range
        vy = vy_mean + 2. * (np.random.random() - .5) * vy_range
        rad = rad_mean + 2. * (np.random.random() - .5) * rad_range
        return vx,vy,y,rad

    camera = Camera(scale=130)
    camera.setY(1.0)
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

    def updataSamplingLearningData(dt, sample, data, state, i, num):
        jumper.update(dt=dt, control=False)
        state[0] = state[1]
        state[1] = jumper.state
        if state[0] == 1 and state[1] != state[0] and i[0] < num:
            sample[3:-1] = [jumper.vx, jumper.vy, jumper.y]
            data[:, i[0]] = sample
            sample_display = [round(v,3) for v in sample]
            print('sample '+str(i[0]+1)+'\tvx0,vy0,y0,vx1,vy1,y1,rad: '+str(sample_display))
            i[0] += 1
            if i[0] == num:
                np.save('mass_spring_learning_data.npy', data)
                print('sample ok')
            vx, vy, y, rad = randomInit()
            jumper.init(0, y, rad, vx, vy)
            sample[:] = [vx, vy, y, 0, 0, 0, rad]
            state[:] = [0,0]

    pyglet.clock.schedule_interval(updataSamplingLearningData,interval=1./100.,
                                   sample=sample, data=data, state=state, i=i, num=num)
    pyglet.app.run()

mass_spring_sampling_data(50)

# Mass spring Learning from learning data
def mass_spring_learning():
    data = np.load('mass_spring_learning_data.npy')
    input = data[:-1, :]
    output = data[-1, :]
    print(input)
    print(output)

# mass_spring_learning()
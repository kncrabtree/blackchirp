#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:11:08 2019

@author: kncrabtree
"""

import blackchirpexperiment as bc
from matplotlib import pyplot as plt

num = 126
path = '.'

x=5
y=5
z=2
t=3300
pStag = .70/.0193368
gamma = 5/3
spacerInches = 3.
zOffset = 191. + 25.
cm = 'cividis'
interp = 'bicubic'

smooth = True
smoothWin = 21
smoothPoly = 3

ex = bc.BlackChirpExperiment(num,path=path)
ms = ex.motorScan
if smooth:
    ms.smooth(smoothWin,smoothPoly)

xv = ms.xVal(x)
yv = ms.yVal(y)
zv = zOffset - ms.zVal(z) + spacerInches*25.4
tv = ms.tVal(t)

z0 = zOffset - ms.z0 + spacerInches*25.4
zMax = zOffset - ms.zMax + spacerInches*25.4

plt.close('all')

plt.figure()
plt.plot(ms.tSlice(x,y,z))
plt.xlabel('T (ns)')
plt.ylabel('Impact Pressure (Torr)')
plt.title('X = {:1.1f} mm, Y = {:1.1f} mm, Z = {:1.1f} mm'.format(xv,yv,zv))


plt.figure()
plt.imshow(ms.xzSlice(y,t),extent=(ms.x0,ms.xMax,z0,zMax),interpolation=interp,origin='lower',cmap=cm)
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
cb = plt.colorbar()
cb.set_label('Impact Pressure (Torr)')
plt.title('Y = {:1.1f} mm, T = {:5.0f} ns'.format(yv,tv))

plt.figure()
plt.imshow(ms.yzSlice(x,t),extent=(ms.y0,ms.yMax,z0,zMax),interpolation=interp,origin='lower',cmap=cm)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
cb = plt.colorbar()
cb.set_label('Impact Pressure (Torr)')
plt.title('X = {:1.1f} mm, T = {:5.0f} ns'.format(xv,tv))

plt.figure()
plt.imshow(ms.xySlice(z,t),extent=(ms.x0,ms.xMax,ms.y0,ms.yMax),interpolation=interp,origin='lower',cmap=cm)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
cb = plt.colorbar()
cb.set_label('Impact Pressure (Torr)')
plt.title('Z = {:1.1f} mm, T = {:5.0f} ns'.format(zv,tv))

plt.figure()
plt.imshow(ms.xtSlice(y,z),extent=(ms.t0,ms.tMax,ms.x0,ms.xMax),interpolation=interp,aspect='auto',origin='lower',cmap=cm)
plt.xlabel('T (ns)')
plt.ylabel('X (mm)')
cb = plt.colorbar()
cb.set_label('Impact Pressure (Torr)')
plt.title('Y = {:1.1f} mm, Z = {:1.1f} mm'.format(yv,zv))

plt.figure()
plt.imshow(ms.ytSlice(x,z),extent=(ms.t0,ms.tMax,ms.y0,ms.yMax),interpolation=interp,aspect='auto',origin='lower',cmap=cm)
plt.xlabel('T (ns)')
plt.ylabel('Y (mm)')
cb = plt.colorbar()
cb.set_label('Impact Pressure (Torr)')
plt.title('X = {:1.1f} mm, Z = {:1.1f} mm'.format(xv,zv))

plt.figure()
plt.imshow(ms.ztSlice(x,y),extent=(ms.t0,ms.tMax,z0,zMax),interpolation=interp,aspect='auto',origin='lower',cmap=cm)
plt.xlabel('T (ns)')
plt.ylabel('Z (mm)')
cb = plt.colorbar()
cb.set_label('Impact Pressure (Torr)')
plt.title('X = {:1.1f} mm, Y = {:1.1f} mm'.format(xv,yv))

plt.figure()
for zz in range(4):
    plt.plot(ms.xPoints(),ms.xSlice(y,zz,t),label='Z = {:1.1f} mm'.format(zOffset - ms.zVal(zz)+ spacerInches*25.4))
plt.legend()

plt.figure()
xdat = ms.zSlice(x,y,t)
xm = []
for pi in xdat:
    mach = ms.machNumber(pi,pStag,gamma)
    xm.append(293/(1+(gamma-1)/2*mach*mach))
plt.plot(zOffset - ms.zPoints() + spacerInches*25.4,xm)

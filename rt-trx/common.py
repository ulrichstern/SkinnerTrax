#
# common for real-time tracker and analysis
#
# 12 Mar 2019 by Ulrich Stern
#

from __future__ import division

import enum

from util import *

# - - -

# chamber type
class CT(enum.Enum):
  regular = dict(ppmm=8.4,
    center=(45, 70))
  htl = dict(ppmm=8.0, nc=5, name='HtL', floor=((0, 0), (80, 128)),
    center=(40, 64))
  large = dict(ppmm=7.0, nc=2, floor=((0, 0), (244, 244)),
    center=(122, 122))

  # returns chamber type given the number of flies per camera
  @staticmethod
  def get(numFlies):
    return {2: CT.regular, 4: CT.large, 20: CT.htl}[numFlies]

  def __str__(self):
    s = self.value.get('name')
    return self.name if s is None else s

  # returns number of pixels per mm at floor of chamber
  def pxPerMmFloor(self): return self.value['ppmm']

  # returns number of chambers per row (in camera view)
  def numCols(self): return self.value.get('nc')

  # returns tl and br frame coordinates for chamber floor for the given fly;
  #  if no fly is given or for regular chamber, tl and br for the full frame
  #  is returned
  def floor(self, xf, f=None):
    if f is None or self is CT.regular:
      return (0, 0), xf.frameSize
    elif self in (CT.htl, CT.large):
      return (xf.t2f(*xy, f=f, noMirr=True) for xy in self.value['floor'])
    else:
      error('not yet implemented')

  # sets (floor) width and height
  def __getattr__(self, name):
    if name in ('width', 'height'):
      self.width, self.height = (self.value['floor'][1][i] for i in (0, 1))
      return getattr(self, name)
    raise AttributeError(name)

  # returns frame coordinates of the center of the chamber floor
  def center(self): return self.value['center']

  @staticmethod
  def _test():
    for ct in CT:
      tlBr = ct.value.get('floor')
      if tlBr:
        test(ct.center, [], np.array(tlBr).mean(axis=0))

# - - -

# coordinate transformer between template and frame coordinates
# * for large and high-throughput chambers, template coordinates are for the
#  top left chamber with 0 representing the top left corner of the floor
class Xformer:

  # tm: dictionary with template match values (keys: fctr, x, and y)
  def __init__(self, tm, ct, frame, fy):
    if tm is not None:
      self.init(tm)
    self.ct, self.nc, self.frameSize = ct, ct.numCols(), imgSize(frame)
    self.fy = fy

  # called explicitly if template match values calculated (the values were not
  #  saved by early versions of rt-trx)
  def init(self, tm):
    self.fctr, self.x, self.y = tm['fctr'], tm['x'], tm['y']

  def initialized(self): return hasattr(self, 'x')

  # shifts template coordinates between top left (TL) and the given fly's
  #  chambers
  def _shift(self, xy, f, toTL):
    if self.ct in (CT.htl, CT.large) and f is not None:
      r, c = divmod(f, self.nc)
      tf = tupleSub if toTL else tupleAdd
      if self.ct is CT.htl:
        xy = tf(xy, (144*c+5, [4, 180, 355, 531][r]))
      else:
        xy = tf(xy, (284*c+4, 284*r+4))
    return xy

  # mirrors template coordinates, with 0 representing top left corner of floor
  def _mirror(self, xy, f, noMirr=False):
    if self.ct in (CT.htl, CT.large) and f is not None:
      r, c = divmod(f, self.nc)
      x, y = xy
      if self.ct is CT.htl:
        xy = (x if c < 3 or noMirr else self.ct.width-x,
          y if r < 2 or not self.fy or noMirr else self.ct.height-y)
      else:
        xy = (x if c == 0 or noMirr else self.ct.width-x, y)
    return xy

  # xforms template coordinates to int-rounded frame coordinates, possibly
  #  changing coordinates for top left chamber (with 0 representing top left
  #  corner of floor) into coordinates for the given fly (used for placing
  #  circles, etc.)
  def t2f(self, x, y, f=None, noMirr=False):
    xy = self._shift(self._mirror((x, y), f, noMirr), f, toTL=False)
    return intR(tupleAdd(tupleMul(xy, self.fctr), (self.x, self.y)))

  # convenience functions
  def t2fX(self, x, f=None): return self.t2f(x, 0, f)[0]
  def t2fY(self, y, f=None): return self.t2f(0, y, f)[1]
  def t2fR(self, r): return intR(r*self.fctr)

  # xforms frame coordinates to template coordinates; x and y can be ndarrays
  def f2t(self, x, y, f=None):
    xy = (x-self.x)/self.fctr, (y-self.y)/self.fctr
    return self._mirror(self._shift(xy, f, toTL=True), f)

# - - -

if __name__ == "__main__":
  print "testing"
  CT._test()
  print "done"


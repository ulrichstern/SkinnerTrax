#
# utilities
#
# 4 Aug 2013 by Ulrich Stern
#

import cv2, matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, math, scipy.stats as st
import cPickle
import os, platform, sys, subprocess, csv, shutil
import re, operator, time, inspect, bisect, collections, hashlib
import itertools, glob, threading
from PIL import Image, ImageFile
from pkg_resources import parse_version
import rdp as rdpPkg, shapely.geometry as sg, inflect

import util

# - - -

DEBUG = False

# OpenCV-style (BGR)
COL_W = 3*(255,)
COL_BK = (0,0,0)
COL_B, COL_B_L, COL_B_D = (255,0,0), (255,64,64), (192,0,0)
COL_G, COL_G_L, COL_G_D = (0,255,0), (64,255,64), (0,192,0)
COL_G_DD, COL_G_D224, COL_G_D96 = (0,128,0), (0,224,0), (0,96,0)
COL_R, COL_R_L, COL_R_D = (0,0,255), (64,64,255), (0,0,192)
COL_Y, COL_Y_D = (0,255,255), (0,192,192)
COL_O = (0,127,255)

JPG_X, AVI_X, AD_AVI_X, TXT_X = (re.compile(p, re.IGNORECASE) for p in
  (r'\.jpg$', r'\.avi$', r'AD\.avi$', r'\.txt$'))
DIGITS_ONLY = re.compile(r'^\d+$')
SIMPLE_FLOAT = re.compile(r'^-?\d+\.\d*$')

SINGLE_UPPERCASE = re.compile(r'([A-Z])')

SPACES_AFTER_TAB = re.compile(r'(?<=\t) +')

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

_PFS = platform.system()
MAC, WINDOWS = _PFS == 'Darwin', _PFS == 'Windows'

OPCV3 = parse_version(cv2.__version__) >= parse_version('3')
CV_AA = cv2.LINE_AA if OPCV3 else cv2.CV_AA

DEVNULL_OUT = open(os.devnull, 'w')
  # note: for devnull, not worth to close file in atexit handler

# - - -

# skip import for certain packages on certain platforms, e.g., due to
#  difficult install (reduces functionality)
if not (WINDOWS or MAC):
  from qimage2ndarray import array2qimage
if not MAC:
  import blist

# - - -

_READ_SIZE = 8192

# - - - exceptions

# base exception
class Error(Exception): pass

# for internal errors
class InternalError(Error): pass
# bad argument(s), CSV data
class ArgumentError(Error): pass
class CsvError(Error): pass
# possible problem with video
class VideoError(Error): pass
# to break out of nested loops
class NestedBreak(Error): pass

# - - - general

# checks that the given function returns the given value
def test(func, args, rval):
  rv = func(*args)
  if not _equal(rv, rval):
    print "args: %s\n  rval: %s\n  correct: %s" %(args, rv, rval)
    raise InternalError()

# for test()
def _equal(v1, v2):
  if isinstance(v1, (list, tuple)):
    return len(v1) == len(v2) and all(_equal(e1, e2) for e1, e2 in zip(v1, v2))
  elif isinstance(v1, (str, slice, dict)) or v1 is None:
    return v1 == v2
  return isClose(v1, v2)

# returns whether the given values (scalars, tuples, etc.) are close
def isClose(v1, v2, atol=1e-08):
  def len1(v): return 1 if np.isscalar(v) else len(v)
  return len1(v1) == len1(v2) and np.allclose(v1, v2, atol=atol, equal_nan=True)

# requires the given values to be close to each other
def requireClose(v1, v2):
  if not isClose(v1, v2):
    print "values not close:\n  v1: %s\n  v2: %s" %(v1, v2)
    raise InternalError()

# prints warning
def warn(msg):
  print "warning: %s" %msg

_NLS_MSG = re.compile(r'^(\n*)(.*)$', re.DOTALL)
# prints message and exits; leading newlines in message are placed before
#  "error"
def error(msg):
  mo = _NLS_MSG.match(msg)
  print "%serror: %s" %(mo.group(1), mo.group(2))
  sys.exit(1)

# prints the given string without newline and flushes stdout
def printF(s):
  sys.stdout.write(s)
  sys.stdout.flush()

# returns file path (joining arguments) if file exists; otherwise, exits
#  with error message or returns False (for noError=True)
# note: typical call: fp = checkIsfile(dir, fn)
def checkIsfile(*fp, **kwargs):
  fp = os.path.join(*fp)
  if not os.path.isfile(fp):
    if kwargs.get('noError'):
      return False
    error("%s does not exist" %fp)
  return fp

# shorthand for checkIsfile(*fp, noError=True)
def isfile(*fp): return checkIsfile(*fp, noError=True)

# returns anonymous object with the given attributes (dictionary)
# note: cannot be pickled
def anonObj(attrs=None):
  return type('', (), {} if attrs is None else attrs)

# returns rounded ints (as tuple if there is more than one) for the given
#  values; can be passed iterable with values
def intR(*val):
  if val and isinstance(val[0], collections.Iterable):
    val = val[0]
  t = tuple(int(round(v)) for v in val)
  return t if len(t) > 1 else t[0]

# returns a+val allowing None for a
def noneadd(a, val): return a if a is None else a+val

# returns a or, iff a is None, val (default: NaN)
def none2val(a, val=np.nan): return val if a is None else a

# returns the distance of two points
def distance(pnt1, pnt2):
  return np.linalg.norm(np.array(pnt1)-pnt2)

# returns preference index or NaN if a + b < n; also works for iterables
def prefIdx(a, b, n=10, amult=1):
  a = np.array(a, np.float)*amult
  s = np.array(a + b)
  s[s < n] = np.nan
  return (a-b)/s

def _prefIdxTest():
  ts = [(0, 10, -1),
    ([10, 15, 5], [0, 5, 4], [1., .5, np.nan])]
  for a, b, pi in ts:
    test(prefIdx, [a, b], pi)

# returns svn revision number with -M appended if modified (or None if svn
#  fails) and file name of the calling module or of the given file
# note: svn can use MD5 or SHA1 for its checksum
_GET_SVN_REV = re.compile(r'^Revision:\s+(\d+)$', re.M)
_GET_SVN_CS = re.compile(r'^Checksum:\s+([0-9a-f]+)$', re.M)
def svnRevision(fn=None):
  if fn is None:
    for frm in inspect.stack()[1:]:   # determine caller
      mdl = inspect.getmodule(frm[0])
      if mdl:
        fn = mdl.__file__
        if fn != __file__:
          break
  try:
    out = executeOutput(["svn", "info", fn])
  except subprocess.CalledProcessError:
    return None, fn
  revs, css = [re.findall(r, out) for r in [_GET_SVN_REV, _GET_SVN_CS]]
  assert len(revs) == 1 and (revs[0] == '0' or len(css) == 1)
  return revs[0] + ('' if not css or css[0] in (md5(fn), sha1(fn))
    else '-M'), fn

# write to both console and file
# usage: sys.stdout = Tee([sys.stdout, logfile])
# name inspired by tee command
class Tee(list):
  def write(self, obj):
    for s in self:
      s.write(obj)
  def flush(self):
    for s in self:
      s.flush()

# simple timer
# note: time.clock() recommended for benchmarking Python, but it excludes
#  time taken by "external" processes on Ubuntu
class Timer():
  def __init__(self, useClock=False):
    self.getTime = time.clock if useClock else time.time
    self.start = self.getTime()
  # gets elapsed time in seconds (as float)
  def get(self, restart=False):
    now = self.getTime()
    elp = now - self.start
    if restart: self.start = now
    return elp
  # gets elapsed time in seconds (as float) and restarts timer
  def getR(self): return self.get(restart=True)
  # restarts the timer
  def restart(self): self.start = self.getTime()

# rate timer for measuring event (e.g., frame) rates
# note: uses either buffer or exponential moving average
class RateTimer():
  def __init__(self, useBuffer=True, bufferSz=20, alpha=0.05):
    self.lts, self.useBuffer = None, useBuffer
    if useBuffer:
      self.dts, self.idx = np.full(bufferSz, np.nan), 0
    else:
      self.dt, self.alp = None, alpha
  # returns rate [1/s] of calls to rate() (None for first call)
  def rate(self):
    ts, avgRt = time.time(), None
    if self.lts is not None:
      dt = ts - self.lts
      if self.useBuffer:
        self.dts[self.idx] = dt
        self.idx = (self.idx+1) % len(self.dts)
        avgRt = 1/np.nanmean(self.dts)
      else:
        self.dt = dt if self.dt is None else (1-self.alp)*self.dt + self.alp*dt
        avgRt = 1/self.dt
    self.lts = ts
    return avgRt

# option overrider
class Overrider():
  def __init__(self, opts):
    self.opts = opts
    self.ovs = []
  # override the given option
  def override(self, name, val=True, descr=None):
    if not hasattr(self.opts, name):
      raise ArgumentError(name)
    descr = descr or SINGLE_UPPERCASE.sub(r' \1', name).lower()
    self.ovs.append((descr, val, getattr(self.opts, name)))
    setattr(self.opts, name, val)
  # report which options were overridden
  def report(self):
    if self.ovs:
      print "overrode options:\n  %s\n" %"\n  ".join(
        '"%s" to %s (from %s)' %e for e in self.ovs)

# executes the given command (e.g., ['gzip', '-f', fn]), raising
#  subprocess.CalledProcessError in case of problems
def execute(cmd, wd=None, logfile=DEVNULL_OUT, pythonDetect=True):
  if pythonDetect and cmd[0] != "python" and cmd[0].endswith(".py"):
    cmd = ["python"] + cmd
  logfile.write("[%s]  %s\n" %(time2str(), " ".join(cmd)))
  subprocess.check_call(cmd, stdout=logfile, stderr=logfile, cwd=wd)

# executed the given command, returning output and raising
#  subprocess.CalledProcessError in case of problems
def executeOutput(cmd):
  if DEBUG:
    print "executing %s" %" ".join(cmd)
  out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
  if DEBUG and out:
    print "| " + "\n| ".join(out.splitlines())
  return out

# returns min and index of min
def minIdx(vals):
  return min(enumerate(vals), key=operator.itemgetter(1))[::-1]
# returns max and index of max
def maxIdx(vals):
  return max(enumerate(vals), key=operator.itemgetter(1))[::-1]

# starts the given callable target as daemon thread
def startDaemon(target, args=()):
  t = threading.Thread(target=target, args=args)
  t.daemon = True
  t.start()

# - - - strings

# converts the given seconds since epoch to YYYY-MM-DD HH:MM:SS format
# notes:
# * current time is used if no seconds since epoch are given
# * use UTC, e.g., to convert seconds to HH:MM:SS (see test)
def time2str(secs=None, format='%Y-%m-%d %H:%M:%S', utc=False):
  if secs is None:
    secs = time.time()
  return time.strftime(format,
    time.gmtime(secs) if utc else time.localtime(secs))

def _time2strTest():
  ts = [([3601, '%H:%M:%S', True], '01:00:01')]
  for args, r in ts:
    test(time2str, args, r)

# converts the given seconds to HH:MM:SS
def s2time(secs): return time2str(secs, '%H:%M:%S', utc=True)

# converts the given frame index to HH:MM:SS
def frame2time(fi, fps): return s2time(fi/fps)
 
# returns the number of seconds for the given time in HH:MM:SS format
def time2s(hms, format='%H:%M:%S'):
  try:
    s = time.strptime(hms, format)
  except ValueError:
    raise ArgumentError(hms)
  return s.tm_hour*3600 + s.tm_min*60 + s.tm_sec

def _time2sTest():
  ts = [('1:01:02', 3662), ('10:00:00', 36000)]
  for hms, s in ts:
    test(time2s, [hms], s)

# returns the numbers of seconds for the given time interval in HH:MM-HH:MM
#  format
# note: uses error() instead of raising ArgumentError
def interval2s(iv):
  try:
    s = [time2s(p, format='%H:%M') for p in iv.split('-')]
  except ArgumentError:
    error('cannot parse times in interval "%s"' %iv)
  if len(s) != 2:
    error('interval "%s" does not have two times separated by "-"' %iv)
  if s[0] >= s[1]:
    error('start time in interval "%s" not before end time' %iv)
  return s

# returns the normalized weights for the given weight string; e.g.,
#  [0.25, 0.75, 0.0] for "1,3,0"
def parseWeights(weights, n=3):
  if weights is None:
    return None
  try:
    ws = [float(w) for w in weights.split(',')]
  except ValueError:
    ws = []
  if len(ws) != n:
    error('cannot parse weights "%s"' %weights)
  t = sum(ws)
  return [w/t for w in ws]

def _parseWeightsTest():
  ts = [(['1,3,0'], [.25, .75, 0.]), (['1.8,.2', 2], [.9, .1])]
  for args, r in ts:
    test(parseWeights, args, r)

# returns list of ints for the given int list string; e.g., [1,2,4] for "1-2,4"
def parseIntList(l):
  ps, ints = l.split(","), []
  for p in ps:
    try:
      fl = [int(i) for i in p.split("-")]
    except ValueError:
      error('cannot parse id list "%s"' %l)
    ints.extend((fl[0],) if len(fl) == 1 else range(fl[0], fl[1]+1))
  return ints

def _parseIntListTest():
  ts = [(['1-2,4'], [1,2,4])]
  for args, r in ts:
    test(parseIntList, args, r)

# removes '/' from end of string if present
def removeSlash(s):
  return s[:-1] if s.endswith('/') else s

# similar to os.path.basename but returns, e.g., 'bar' for '/foo/bar/'
def basename(s, withExt=True):
  bn = os.path.basename(removeSlash(s))
  return bn if withExt else os.path.splitext(bn)[0]

# adds prefix to filename (base name)
def addPrefix(path, prefix):
  dn, bn = os.path.split(path)
  return os.path.join(dn, prefix+bn)

# returns plural "s"
def pluralS(n):
  return "" if n == 1 else "s"
_INFL = inflect.engine()
# returns "n items" with proper plural
def nItems(n, item):
  return "%d %s" %(n, item if n == 1 else _INFL.plural(item))

# joins, e.g., list of ints or floats (using the given precision)
def join(withStr, l, lim=None, p=None, end=False):
  if lim and len(l) > lim:
    l = l[-lim:] if end else l[:lim]
    l.insert(0 if end else len(l), "...")
  ff = "" if p is None else "%%.%df" %p
  return withStr.join(
    ff %e if ff and isinstance(e, float) else str(e) for e in l)

# joins the given list using "," and "and"
def commaAndJoin(l):
  return join(" and ", l) if len(l) < 3 else \
    join(", ", l[:-1]) + ", and %s" %l[-1]

def _commaAndJoinTest():
  ts = [([1], "1"), ([1, 2], "1 and 2"), ([1, 2, 3], "1, 2, and 3")]
  for l, r in ts:
    test(commaAndJoin, [l], r)

# replaces the given pattern with the given replacement and checks that
#  the resulting string is different from original
def replaceCheck(pattern, repl, string):
  s = re.sub(pattern, repl, string)
  if s == string:
    raise ArgumentError(s)
  return s

def _replaceCheckTest():
  try:
    replaceCheck(AVI_X, ".bar", "foo")
  except ArgumentError:
    pass
  test(replaceCheck, [AVI_X, ".bar", "foo.avi"], "foo.bar")

# returns the first subgroup and checks that there is a match
def firstGroup(pattern, s, flags=0):
  mo = re.search(pattern, s, flags)
  if not mo:
    raise ArgumentError(s)
  return mo.group(1)

# matches pattern against the given list of strings and returns list with
#  first subgroups
def multiMatch(pattern, l):
  mos = (re.match(pattern, s) for s in l)
  return [mo.group(1) for mo in mos if mo]

def _multiMatchTest():
  ts = [([r'v(\d+)$', ['v1', ' v2', 'v3 ', 'v44']], ['1', '44'])]
  for args, r in ts:
    test(multiMatch, args, r)

# converts float to string using up to the given precision, skipping trailing
#  zeros
def formatFloat(f, p):
  return (("%%.%df" %p) %f).rstrip('0').rstrip('.')

def _formatFloatTest():
  ts = [((1, 1), "1"), ((1.11, 1), "1.1"), ((1.16, 1), "1.2")]
  for args, r in ts:
    test(formatFloat, args, r)

# converts string to int or float depending on value
# notes:
# * currently not supported: e.g., "-1", ".2"
# * alternative: try int() and catch ValueError etc.
def toNumeric(s, noFloat=False):
  if isinstance(s, str):
    if DIGITS_ONLY.match(s): return int(s)
    elif not noFloat and SIMPLE_FLOAT.match(s): return float(s)
  return s

# converts string to int depending on value
def toInt(s): return toNumeric(s, noFloat=True)

# determines repeats in the given list of string, returning, e.g.,
#  [("a", 2), ("b", 1), ("a", 1)] for ["a", "a", "b", "a"]
def repeats(l):
  r, c = [], 0
  for i, e in enumerate(l):
    c = c + 1
    if i == len(l)-1 or e != l[i+1]:
      r.append((e, c))
      c = 0
  return r

def _repeatsTest():
  ts = [(["a", "a", "b", "a"], [("a", 2), ("b", 1), ("a", 1)])]
  for l, r in ts:
    test(repeats, [l], r)

# - - - lists

# returns duplicates for the given list, returning, e.g., [2] for [1, 2, 2]
def duplicates(l):
  return [e for e, cnt in collections.Counter(l).items() if cnt > 1]

def _duplicatesTest():
  ts = [([1, 2, 2], [2]), ([1, 2], [])]
  for l, r in ts:
    test(duplicates, [l], r)

# concatenates lists in the given list of lists, returning, e.g., [1, 2, 3] for
#  [[1, 2], [3]]
def concat(l, asIt=False):
  it = itertools.chain.from_iterable(l)
  return it if asIt else list(it)

# - - - tuples

# returns t2 as tuple
#  if t2 is int, float, or string, t2 is replicated len(t1) times
#  otherwise, t2 is passed through
def _toTuple(t1, t2):
  if isinstance(t2, (int,float,str)):
    t2 = len(t1) * (t2,)
  return t2

# applies the given operation to the given tuples; t2 can also be number
def tupleOp(op, t1, t2):
  return tuple(map(op, t1, _toTuple(t1, t2)))

# tupleOp() add
def tupleAdd(t1, t2): return tupleOp(operator.add, t1, t2)

# tupleOp() subtract
def tupleSub(t1, t2): return tupleOp(operator.sub, t1, t2)

# tupleOp() multiply
def tupleMul(t1, t2): return tupleOp(operator.mul, t1, t2)

# - - - blist

# returns the n elements before and after the given object in the given
#  sorted list or blist.sortedlist
def beforeAfter(sl, obj, n=1):
  blsl = isinstance(sl, blist.sortedlist)
  li = sl.bisect_left(obj) if blsl else bisect.bisect_left(sl, obj)
  ri = sl.bisect(obj) if blsl else bisect.bisect(sl, obj)
  res = 2*n*[None]
  for i in range(-n, n):
    i1 = li+i if i < 0 else ri+i
    res[i+n] = None if i1 < 0 or i1 >= len(sl) else sl[i1]
  return res

def _beforeAfterTest():
  l = [1, 4, 4, 6, 10]
  d = {
    4: [1, 6], 1: [None, 4], 0: [None, 1], 8: [6, 10], 20: [10, None],
    (4, 2): [None, 1, 6, 10], (6, 3): [1, 4, 4, 10, None, None] }
  for i in [0, 1]:
    if i == 1: l = blist.sortedlist(l)
    for obj, rv in d.iteritems():
      args = [l] + list(obj) if isinstance(obj, tuple) else [l, obj]
      test(beforeAfter, args, rv)

# - - - numpy

# returns slice objects for all contiguous true regions in the given array
def trueRegions(a):
  r = np.ma.flatnotmasked_contiguous(np.ma.array(a, mask=~a))
  return [] if r is None else r

def _trueRegionsTest():
  ts = [
    (np.array([True,False,True,True]), [slice(0,1), slice(2,4)]),
    (np.array([False]), [])]
  for a, r in ts:
    test(trueRegions, [a], r)

# returns array "reduced" to the given range [r1, r2) (i.e., array with only
#  those elements that fall into range), or counts elements in range
def inRange(a, r1, r2, count=False):
  idxs = (r1 <= a) & (a < r2)
  return np.count_nonzero(idxs) if count else a[idxs]

# returns min and max
def minMax(a): return a.min(), a.max()

# matches the elements in array a2 with those in (longer) array a1,
#  returning Boolean array of length len(a1) with True for matches and
#  whether all elements in a2 were matched
# note: not super-efficient
def matchArrays(a1, a2, incr=True):
  if not incr:
    if isinstance(a1, (list, tuple)):
      a1, a2 = a1[::-1], a2[::-1]
    else:
      a1, a2 = a1[::-1, ...], a2[::-1, ...]
  l1, l2 = len(a1), len(a2)
  i2, m = 0, np.zeros((l1,), dtype=bool)
  for i1 in range(l1):
    if np.array_equal(a1[i1], a2[i2]):
      m[i1] = True
      i2 += 1
      if i2 == l2:
        break
  return m if incr else m[::-1], i2 == l2

def _matchArraysTest():
  ts = [(([1, 0, 3], [1, 3]), ([True, False, True], True)),
    (([1, 0, 3], [0, 2]), ([False, True, False], False)),
    (([1, 0], [1], False), ([True, False], True))]
  for args, r in ts:
    test(matchArrays, args, r)

# - - - geometry

# common parameters
# xy: sequence of points (or trajectory) in one of two formats:
#  - tuple with x and y arrays
#  - matrix with point in each row

# converts points to matrix format
def xy2M(xy):
  return np.array(xy).T if isinstance(xy, tuple) else xy

# converts points to tuple format
def xy2T(xy):
  return xy if isinstance(xy, tuple) else (xy[:,0], xy[:,1])

# returns the distances between the given points or from the first given point
#  to each point
# note: see also distance()
def distances(xy, fromFirst=False):
  x, y = xy2T(xy)
  dx, dy = (x-x[0], y-y[0]) if fromFirst else (np.diff(x), np.diff(y))
  return np.linalg.norm([dx, dy], axis=0)

# returns the velocity angles for the given points; empty array is returned
#  if there are not enough points
# note: returns values from -pi to pi, see np.arctan2() for details
def velocityAngles(xy):
  x, y = xy2T(xy)
  return np.arctan2(np.diff(y), np.diff(x))

# normalizes the given angles to be between -pi and pi
def normAngles(a):
  return np.mod(a+np.pi, 2*np.pi) - np.pi

# returns the turn angles for the given points; empty array is returned if
#  there are not enough points
# note: n points yield n-2 turn angles
def turnAngles(xy):
  return normAngles(np.diff(velocityAngles(xy)))

def _turnAnglesTest():
  ts = [([0,0,-1], [0,1,0], 135), ([0,1,2], [0,0,1], 45),
    ([0,-1,-2], [0,0,1], -45), ([0,-1,-1], [0,-1,0], -135),
    ([], [], None), ([0], [0], None), ([0,0], [0,1], None)]
  for x, y, r in ts:
    test(turnAngles, [(x, y)], [] if r is None else r*np.pi/180)

# returns whether the given point is inside the given rectangle plus the given
#  border
def inRect(xy, tlbr, bw=0):
  xm, ym, xM, yM = tlbr
  if xm > xM: xm, xM = xM, xm
  if ym > yM: ym, yM = yM, ym
  return xm-bw < xy[0] < xM+bw and ym-bw < xy[1] < yM+bw

_TEST_RDP = False

# returns RDP-simplified trajectory in matrix format for the given points and
#  indexes of points kept
def rdp(xy, epsilon, useRdpPkg=False):
  xy = xy2M(xy)
  if useRdpPkg:
    error("points kept not yet implemented")
    return rdpPkg.rdp(xy, epsilon=epsilon)
  else:
    # Shapely's simplify() is fast but does not return the points kept
    # solution: use small z values to maintain point index info
    if len(xy) == 1:
      return xy, np.array([0])
    else:
      fctr = 1e20
      xy = np.hstack((xy, (1/fctr)*np.arange(len(xy))[:, None]))
      def getIdxs(xyi): return (fctr*xyi[:,2]+.5).astype(int)
      sxy = np.array(sg.LineString(xy).simplify(epsilon,
        preserve_topology=False))
      si = getIdxs(sxy)
      if _TEST_RDP:
        assert np.array_equal(getIdxs(xy), np.arange(len(xy)))
        assert si[0] == 0, si[-1] == len(xy)-1
      return sxy[:,:2], si

# - - - matplotlib

# plots text raising ArgumentError for NaN positions, avoiding "cryptic" error:
#  https://github.com/matplotlib/matplotlib/issues/4318#issuecomment-436266884
def pltText(x, y, s, **kwargs):
  if np.isnan(x + y):
    raise ArgumentError(x, y)
  return plt.text(x, y, s, **kwargs)

# - - - stats

# returns mean, confidence interval, and number of finite values for the
#  given array
# notes:
# * confidence interval matches what R gives for t.test(a)
# * SEM multiplier matches GraphPad's "confidence interval of a mean" page
def meanConfInt(a, conf=0.95, asDelta=False):
  a = np.array(a)
  a = a[np.isfinite(a)]
  n = len(a)
  mean = np.mean(a) if n else np.nan
  d = st.t.ppf((1+conf)/2., n-1) * st.sem(a) if n > 1 else np.nan
  return (mean, d, n) if asDelta else (mean, mean - d, mean + d, n)

# returns 'ns', '*', ... for the given P value, nanR if P is NaN
# note: follows https://graphpad.com/support/faqid/978/
def p2stars(p, nsWithP=False, nanR=None):
  return nanR if np.isnan(p) else (
    '****' if p <= 0.0001 else ('***' if p <= 0.001 else (
    '**' if p <= 0.01 else ('*' if p <= 0.05 else
    'ns' + (' (p=%.2f)' %p if nsWithP else '')))))

# - - - dictionary

# inverts mapping; if mapping has non-unique values, possibly use toSet
def invert(m, toSet=False):
  it = m.iteritems() if isinstance(m, dict) else iter(m)
  if toSet:
    d = collections.defaultdict(set)
    for k, v in it:
      d[v].add(k)
    return d
  else:
    return dict((v, k) for k, v in it)

def _invertTest():
  ms = [
    [{1:1, 2:2, 3:1}, True, {1:set([1,3]), 2:set([2])}],
    [[(1,4),(2,5)], False, {4:1, 5:2}] ]
  for m, toSet, invM in ms:
    test(invert, [m, toSet], invM)

# - - - file

# returns absolute path for file that is part of package given relative name
def packageFilePath(fn):
  return os.path.join(MODULE_DIR, fn)

# returns content of given file; None if file does not exist
def readFile(fn):
  if os.path.isfile(fn):
    with open(fn, 'rb') as f:
      return f.read()
  return None

# backs up (by renaming or copying) the given file
def backup(fn, verbose=False, copy=False):
  if os.path.isfile(fn):
    if verbose:
      print "backing up %s" %os.path.basename(fn)
    fn1 = fn+'.1'
    if copy:
      shutil.copyfile(fn, fn1)
    else:
      if os.path.isfile(fn1):
        os.remove(fn1)
      os.rename(fn, fn1)

# saves the given object in the given file, possibly creating backup
def pickle(obj, fn, backup=False, verbose=''):
  if backup:
    util.backup(fn, 'B' in verbose)
  f = open(fn, 'wb')
  cPickle.dump(obj, f, -1)
  f.close()

# loads object from the given file
def unpickle(fn):
  if not os.path.isfile(fn):
    return None
  f = open(fn, 'rb')
  obj = cPickle.load(f)
  f.close()
  return obj

# returns secure hash of the given file
def hash(fn, tp='md5'):
  m = getattr(hashlib, tp)()
  with open(fn, 'rb') as f:
    while True:
      data = f.read(_READ_SIZE)
      if not data: break
      m.update(data)
    return m.hexdigest()

# returns MD5 or SHA1 of the given file
def md5(fn): return hash(fn)
def sha1(fn): return hash(fn, 'sha1')

# gzips the given file, overwriting a possibly existing ".gz" file
def gzip(fn):
  execute(['gzip', '-f', fn])

# reads CSV file skipping comment rows
def readCsv(fn, toInt=True, nCols=None):
  if not os.path.isfile(fn): return None
  with open(fn) as f:
    dt = list(csv.reader(row for row in
      (row.partition('#')[0].strip() for row in f) if row))
    if nCols:
      for row in dt:
        if len(row) != nCols:
          raise CsvError("nCols=%d, row=%s" %(nCols, "|".join(row)))
    return [map(int, row) for row in dt] if toInt else dt

_GLOB_CHARS = re.compile(r'[*?[]')

# returns list of filenames given filename or directory name and pattern (to
#  filter directory); a comma-separated list of names can also be given;
#  names can contain wildcards *, ?, and []
def fileList(fn, op=None, tp="videos", pattern=AD_AVI_X):
  if "," in fn:
    return concat(fileList(f, op, tp, pattern) for f in fn.split(","))
  isd = os.path.isdir(fn)
  if isd or _GLOB_CHARS.search(fn):
    fns = [(os.path.join(fn, f) if isd else f)
      for f in (os.listdir(fn) if isd else glob.glob(fn))
      if re.search(pattern, f)]
    if not fns and op:
      print "directory %s has no %s to %s" %(fn, tp, op)
        # TODO: basename(fn) that has subdirectory and file
  else:
    checkIsfile(fn)
    fns = [fn]
  return fns

# writes the command line arguments to the given log file
def writeCommand(lf, csvStyle=False, inclRev=True):
  cmd = '# command: %s' %' '.join(sys.argv)
  if inclRev:
    rev = svnRevision(sys.argv[0])[0]
    cmd += "  [%s]" %("rev. unknown" if rev is None else "r%s" %rev)
  if csvStyle:
    csv.writer(lf, lineterminator='\n\n').writerow((cmd,))
  else:
    lf.write(cmd + '\n\n')

# - - - OpenCV

# turns the given x and y arrays or the given list of alternating x and y
#  values into array with array of points for, e.g., fillPoly() or polylines()
def xy2Pts(*xy):
  if len(xy) == 2 and isinstance(xy[0], collections.Iterable):
    pts = np.array(xy).T
  else:
    pts = np.array(xy).reshape(-1, 2)
  return np.int32(pts[None] + .5)

# returns image; if color value is integer, it is used for all channels
# note: cv2.getTextSize() uses order w, h ("x first")
def getImg(h, w, nCh=3, color=255):
  img = np.zeros(intR((h, w, nCh) if nCh > 1 else (h, w)), np.uint8)
  if isinstance(color, tuple):
    img[:,:] = color
  elif color:
    img[...] = color
  return img

# shows image, possibly resizing window
def imshow(winName, img, resizeFctr=None, maxH=1000):
  h, w = img.shape[:2]
  h1 = None
  if resizeFctr is None and h > maxH:
    resizeFctr = float(maxH)/h
  if resizeFctr is not None:
    h1, w1 = intR(h*resizeFctr, w*resizeFctr)
    if MAC:
      img = cv2.resize(img, (0,0), fx=resizeFctr, fy=resizeFctr)
    else:
      cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
  cv2.imshow(winName, img)
  if h1 and not MAC:
    cv2.resizeWindow(winName, w1, h1)

# min max normalizes the given image
def normalize(img, min=0, max=255):
  return cv2.normalize(img, None, min, max, cv2.NORM_MINMAX, -1)

# shows normalized image
def showNormImg(winName, img):
  imshow(winName, normalize(img, max=1))

# converts the given image to gray
def toGray(img):
  return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# returns gray image using the given channel; if channel is 3, same as toGray()
def toChannel(img, channel):
  return toGray(img) if channel == 3 else img[:,:,channel]

# converts the given image to color (BGR)
# note: possibly include conversion to np.uint8
def toColor(img):
  return img if numChannels(img) > 1 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# returns whether the given image is grayscale
def isGray(img):
  return numChannels(img) == 1 or \
    np.array_equal(img[:,:,0], img[:,:,1]) and \
    np.array_equal(img[:,:,0], img[:,:,2])

# converts the given image to the given type, allowing None image
def astype(img, dtype, rint=False):
  return img if img is None or img.dtype == dtype else \
    (np.rint(img) if rint else img).astype(dtype)

# returns rectangular subimage, allowing tl and br points outside the image
# note: tl is included, br is excluded
def subimage(img, tl, br):
  h, w = img.shape[:2]
  tlx, tly = tl
  brx, bry = br
  return img[max(tly,0):min(bry,h), max(tlx,0):min(brx,w)]

# returns the image overlayed with the given mask in the given color
#  (mask values: 0: keep image ... 255: use color)
def overlay(img, mask, color=COL_G):
  imgC = pilImage(toColor(img))
  imgC.paste(color[::-1], None, pilImage(astype(mask, np.uint8, True)))
    # note: alternative: cv2.addWeighted()
  return cvImage(imgC)

# extends the given image
# note: possibly easier: use cv2.copyMakeBorder()
def extendImg(img, trbl, color=255):
  h, w = img.shape[:2]
  x, y = trbl[3], trbl[0]
  imgE = getImg(y + h + trbl[2], x + w + trbl[1], nCh=numChannels(img),
    color=color)
  imgE[y:y+h, x:x+w] = img
  return imgE

# returns the bottom right point of the given image
def bottomRight(img):
  return (img.shape[1]-1, img.shape[0]-1)

# returns image rotated by the given angle (in degrees, counterclockwise)
def rotateImg(img, angle):
  cntr = tupleMul(img.shape[:2], .5)
  mat = cv2.getRotationMatrix2D(cntr, angle, 1.)
  return cv2.warpAffine(img, mat, img.shape[:2], flags=cv2.INTER_LINEAR)

# returns the number of channels in the given image
def numChannels(img): return img.shape[2] if img.ndim > 2 else 1

# returns the image size as tuple (width, height)
def imgSize(img): return img.shape[1::-1]

# returns area of the given Box2D tuple ((x,y), (w,h), theta)
def boxArea(box): return box[1][0]*box[1][1]

# creates large image out of small ones
# params: imgs: list of images or of (image, header text) tuples
#  nc: number columns, d: distance between small images,
#  hdrs: header text, style: for text, hd: header distance,
#  adjustHS: whether to adjust horizontal spacing to fit headers
#  resizeFctr: applied to imgs, hdrL: header for large image
# returns: large image, list of tuples with positions (tl) of small images
# notes:
# * images that are None result in empty space
def combineImgs(imgs, nc=10, d=10, hdrs=None, style=None, hd=3, adjustHS=True,
    resizeFctr=None, hdrL=None):
  if isinstance(imgs[0], tuple):
    assert hdrs is None
    imgs, hdrs = zip(*imgs)
  style = style or textStyle()
  if resizeFctr is not None:
    imgs = [cv2.resize(img, (0,0), fx=resizeFctr, fy=resizeFctr)
      for img in imgs]
  nnimgs = [img for img in imgs if img is not None]
  h, w = (max(img.shape[i] for img in nnimgs) for i in (0, 1))
  nCh = max(numChannels(img) for img in nnimgs)
  hdrH, hd = (textSize(hdrs[0], style)[1], hd) if hdrs else (0, 0)
  wA = max(w, max(textSize(hdr, style)[0] for hdr in hdrs)) \
    if adjustHS and hdrs else w
  hdrLH, hdrLD = (textSize(hdrL, style)[1], d) if hdrL else (0, 0)
  nr, nc = math.ceil(float(len(imgs))/nc), min(len(imgs), nc)
  hL, wL = hdrLH+hdrLD + nr*(h+d+hdrH+hd), nc*(wA+d)
  imgL = getImg(hL, wL, nCh)
  if hdrL:
    putText(imgL, hdrL, (d/2, d/2+hdrLH), (0,0), style)
  xyL, ndL = len(imgs)*[None], imgL.ndim
  for i, img in enumerate(imgs):
    r, c = divmod(i, nc)
    xL, yL = d/2 + c*(wA+d), d/2+hdrLH+hdrLD+hdrH+hd + r*(h+d+hdrH+hd)
    if img is not None:
      h1, w1 = img.shape[:2]
      imgL[yL:yL+h1, xL:xL+w1] = img if img.ndim == ndL else img[..., None]
    if hdrs and hdrs[i]:
      putText(imgL, hdrs[i], (xL, yL-hd), (0,0), style)
    xyL[i] = (xL, yL)
  return imgL, xyL

# returns the median of the given image
def median(img):
  return np.median(img.copy())

# returns heatmap (image) for the given map (containing counts or probabilities)
# notes:
# * mapCheck: if map contains 0s and np.log is used, map is incremented by 1
# * intermediate check-in
def heatmap(mp, xform=np.log, xySeps=None, mapCheck=False, thres=None):
  xySeps = xySeps or ([], [])
  h, w = mp.shape[:2]
  imgL = getImg(h, w)
  if mapCheck:
    mpm = np.amin(mp)
    if mpm < 0:
      raise ArgumentError(mpm)
    elif mpm == 0 and xform is np.log:
      mp = mp + 1
  x1, ys = 0, xySeps[1]+[h]
  for x2 in xySeps[0]+[w]:
    y1 = 0
    for y2 in ys:
      m = mp[y1:y2, x1:x2]
      # TODO: experiment with thresholding
      if thres is not None:
        msz = m.size
        msm = np.sum(m) - (np.amin(m) == 1)*msz
        thv = thres * msm/float(msz)
        m = cv2.threshold(m, thv, thv, cv2.THRESH_TRUNC)[1]
      if xform is not None:
        m = xform(m)
      img = normalize(m).astype(np.uint8)
      imgL[y1:y2, x1:x2] = cv2.applyColorMap(img, cv2.COLORMAP_JET)
      y1 = y2
    x1 = x2
  return imgL

# returns Matplotlib colormap for the given OpenCV colormap
def mplColormap(cmap=cv2.COLORMAP_JET):
  cols = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cmap)
  return mpl.colors.ListedColormap(cols[:,0,::-1]/255.)

# alpha-blends (with white background) the given colormap
def alphaBlend(cm, alpha):
  rgba = cm(np.arange(cm.N))
  rgba[:,:3] = alpha*rgba[:,:3] + (1-alpha)
  return mpl.colors.ListedColormap(rgba)

# returns Canny edge image and fraction of non-black pixels (before post blur)
def edgeImg(img, thr1=20, thr2=100, preBlur=True, postBlur=True, preNorm=False):
  img = toGray(img)
  if preNorm:
    img = normalize(img)
  if preBlur:
    img = cv2.GaussianBlur(img, (3, 3), 0)
  img = cv2.Canny(img, thr1, thr2)
  nz, npx = img.nonzero()[0].size, img.shape[0]*img.shape[1]
  if postBlur:
    img = cv2.GaussianBlur(img, (3, 3), 0)
  return img, float(nz)/npx

# matches the given template(s) against the given image(s)
#  e.g., (img, tmpl, img2, tmpl2, 0.5)
#    note: second match is weighted with factor 0.5 (default: 1)
# returns result image, top left x, top left y, bottom right (as tuple),
#  minimum distance between template and image border, match value,
#  and non-normalized match values
def matchTemplate(img, tmpl, *args):
  imgs, tmpls, fctrs = [img], [tmpl], [1]
  idx = 0
  for arg in args:
    if isinstance(arg, (int,float)):
      fctrs[idx] = arg
    else:
      if idx > len(tmpls)-1:
        tmpls.append(arg)
      else:
        imgs.append(arg)
        fctrs.append(1)
        idx += 1
  res, maxVals = 0, []
  for i, t, f in zip(imgs, tmpls, fctrs):
    r = cv2.matchTemplate(i, t, cv2.TM_CCOEFF_NORMED)
    maxVals.append(cv2.minMaxLoc(r)[1])
    if len(imgs) > 1:
      r = normalize(r, max=1) * f
    res += r
  minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
  tlx, tly = maxLoc
  br = (tlx+tmpl.shape[1], tly+tmpl.shape[0])
  minD = min(min(maxLoc), img.shape[1]-br[0], img.shape[0]-br[1])
  return res, tlx, tly, br, minD, maxVal, maxVals

# returns the normalized correlation of the given images
# note: values range from -1 to 1 (for identical images)
def normCorr(img1, img2):
  assert img1.shape == img2.shape
  r = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
  assert r.shape == (1,1)
  return r[0,0]

# returns the default text style
def textStyle(size=.9, color=COL_BK):
  return (cv2.FONT_HERSHEY_PLAIN, size, color, 1, CV_AA)

# returns tuple with text width, height, and baseline; style is list with
#  putText() args fontFace, fontScale, color, thickness, ...
# note: 'o', 'g', and 'l' are identical in height and baseline
def textSize(txt, style):
  wh, baseline = cv2.getTextSize(txt, *[style[i] for i in [0,1,3]])
  return wh + (baseline,)

# puts the given text on the given image; whAdjust adjusts the text position
#  using text width and height (e.g., (-1, 0) subtracts the width);
#  text can contain \n-separated lines and \t-separated columns
# note: pos gives bottom left corner of text
def putText(img, txt, pos, whAdjust, style, lnHeightMult=1.7, colWidthMult=1.1):
  cs, ws = [], []
  for ln in (txt.splitlines()):
    cs1 = ln.split('\t')
    cs.append(cs1)
    ws.append([textSize(c, style)[0] for c in cs1])
  ncols = max(len(cs1) for cs1 in cs)
  wh = textSize(cs[0][0], style)[:2]   # first cell currently used for whAdjust
  pos = intR(tupleAdd(pos, tupleMul(whAdjust, wh)))
  for c in range(ncols):
    for i, cs1 in enumerate(cs):
      if c < len(cs1):
        cv2.putText(img, cs1[c], intR(tupleAdd(pos, (0, i*wh[1]*lnHeightMult))),
          *style)
    colw = max(ws1[c] if c < len(ws1) else 0 for ws1 in ws)
    pos = intR(tupleAdd(pos, (colw*colWidthMult, 0)))

_PROPS_WH = ("FRAME_WIDTH", "FRAME_HEIGHT")
# returns new VideoCapture given filename or device number and checks whether
#  the constructor succeeded
def videoCapture(fnDev):
  cap = cv2.VideoCapture(toInt(fnDev))
  if not cap.isOpened():
    raise VideoError("could not open VideoCapture for %s" %fnDev)
  return cap
# returns VideoCapture property id given, e.g., "FPS"
def capPropId(prop):
  return prop if isinstance(prop, int) else \
    getattr(cv2 if OPCV3 else cv2.cv,
      ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)
# returns or sets the given property (e.g., "FPS") for the given VideoCapture
def capProp(cap, prop, val=None, getInt=True):
  if val is None:
    val = cap.get(capPropId(prop))   # cap.get() returns float
    return int(val) if getInt else val
  else:
    cap.set(capPropId(prop), val)
# returns FOURCC for the given VideoCapture (e.g., 'MJPG')
def fourcc(cap):
  fcc = capProp(cap, "FOURCC")
  return "".join(chr(fcc >> s & 0xff) for s in range(0,32,8))
# returns or sets frame rate for the given VideoCapture (e.g., 7.5)
def frameRate(cap, fps=None):
  return capProp(cap, "FPS", fps, getInt=False)
# returns frame count for the given VideoCapture
def frameCount(cap):
  return capProp(cap, "FRAME_COUNT")
# returns frame height for the given VideoCapture
def frameHeight(cap):
  return readFrame(cap, 0).shape[0] if OPCV3 else capProp(cap, "FRAME_HEIGHT")
    # note: FRAME_HEIGHT failed on MJPG for OpenCV 3.0
# returns or sets frame size (width, height) for the given VideoCapture
def frameSize(cap, wh=None):
  if wh is None:
    return tuple(capProp(cap, p) for p in _PROPS_WH)
  if frameSize(cap) != wh:
    for p, v in zip(_PROPS_WH, wh):
      capProp(cap, p, v)
    if frameSize(cap) != wh:
      raise VideoError("cannot set camera resolution to %dx%d" %wh)

# returns the given frame for the given VideoCapture
def readFrame(cap, n):
  def nf(): raise VideoError("no frame %d" %n)
  if n < 0: nf()
  setPosFrame(cap, n)
  ret, frm = cap.read()
  if not ret: nf()
  return frm
# sets the current position for the given VideoCapture
def setPosFrame(cap, n=0):
  capProp(cap, "POS_FRAMES", n)

# returns FOURCC value for OpenCV given, e.g., 'MJPG'
def cvFourcc(fcc):
  return cv2.VideoWriter_fourcc(*fcc) if OPCV3 else cv2.cv.CV_FOURCC(*fcc)

# returns frame range for the given VideoCapture, possibly limited to the
#  given interval in HH:MM-HH:MM format
def frameRange(cap, interval=None):
  nf = frameCount(cap)
  if interval is None:
    return 0, nf
  fps = frameRate(cap)
  return intR(min(nf, s*fps) for s in interval2s(interval))

# returns PIL image given OpenCV image
def pilImage(img):
  return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if
    numChannels(img) > 1 else img)

# returns OpenCV image given PIL image
def cvImage(img):
  if isinstance(img, (Image.Image, ImageFile.ImageFile)):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  raise ArgumentError("not implemented for %s" %type(img))

# reads string from keyboard
# note: currently not used
def readStringKeyboard():
  s = ""
  while True:
    k = cv2.waitKey(0)
    if k == 13:   # Return
      return s
    elif k < 128:   # ignore, e.g., Shift
      s += chr(k)

# - - - Qt

# returns QImage given OpenCV image
def qimage(img):
  return array2qimage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# - - - V4L / V4L2

_V4L2CTL = "v4l2-ctl"
_GET_CTRL_DFLTS = re.compile(r'^\s*(\w+) .*? default=(\d+) value=(\d+)', re.M)
_GET_BUS_INFO = re.compile(r'^\s*Bus info\s*:\s*(.*?)\s*$', re.M)
_GET_BUS_INFOS = re.compile(r' \((.*?)\):\s*$\s*/dev/video(\d+)\s*$', re.M)
# sample output parsed with these regular expressions:
'''
NZXT-U:rt-trx> v4l2-ctl -d 0 --list-ctrls
             brightness (int)    : min=30 max=255 step=1 default=133 value=133
          exposure_auto (menu)   : min=0 max=3 default=1 value=1
...

NZXT-U:rt-trx> v4l2-ctl -d 0 -D
Driver Info (not using libv4l2):
	Driver name   : uvcvideo
	Card type     : Microsoft LifeCam Cinema(TM)
	Bus info      : usb-0000:00:1a.0-1.3
	Driver version: 3.5.7
...

NZXT-U:rt-trx> v4l2-ctl --list-devices
Microsoft LifeCam Cinema(TM) (usb-0000:00:14.0-1):
	/dev/video1

Microsoft LifeCam Cinema(TM) (usb-0000:00:1a.0-1.3):
	/dev/video0
...
'''

# gets or sets V4L2 control(s) for the given device, or if ctl is None, returns
#  default and value for each control (e.g., [("focus_auto", "0", "0"), ...]);
#  to set controls in batches, separate them with None in the list of controls;
#  raises subprocess.CalledProcessError in case of problems
# ctl param examples:
#  "focus_auto" to get
#  ("focus_auto", 0) or [("focus_auto", 0), ...] to set
def v4l2Control(dev, ctl=None, defaultOtherCtls=False):
  getCtl, getDflts = isinstance(ctl, str), ctl is None
  if not (getCtl or getDflts):
    if isinstance(ctl, tuple):
      ctl = [ctl]
    if not isinstance(ctl, list):
      raise ArgumentError(ctl)
    if defaultOtherCtls:
      v4l2Control(dev, ctl)
        # set non-defaults first to avoid v4l2-ctl errors if, e.g., focus_auto
        # and focus_absolute get set in the same call while focus_auto is on
      dflts = v4l2Control(dev)
      ctlSet = set(cv[0] for cv in ctl if cv is not None)
      ctl = [(c, d) for (c, d, v) in dflts if c not in ctlSet and v != d]
    if None in ctl:
      ni = ctl.index(None)
      v4l2Control(dev, ctl[:ni])
      v4l2Control(dev, ctl[ni+1:])
      return
    ctl = [(c, v) for (c, v) in ctl if c is not None]
    if not ctl:
      return
  cmd = [_V4L2CTL, "-d", str(dev)]
  cmd.extend(["--list-ctrls"] if getDflts else (
    ["-C", ctl] if getCtl else
    ["-c", ",".join("%s=%s" %cv for cv in ctl)]))
  if False:   # debugging; alternative: DEBUG
    print cmd
  out = executeOutput(cmd)
  if getCtl:
    val = firstGroup(r'^%s:\s+(.*?)\s*$' %ctl, out, re.M)
    return int(val) if re.match(DIGITS_ONLY, val) else val
  elif getDflts:
    dflts = re.findall(_GET_CTRL_DFLTS, out)
    assert len(dflts)
    return dflts

# returns "bus info" for the given device (e.g., "usb-0000:00:14.0-1") or
#  map from bus info to device number for all connected devices, raising
#  subprocess.CalledProcessError in case of problems
def v4l2BusInfo(dev=None):
  all, cmd = dev is None, [_V4L2CTL]
  cmd.extend(["--list-devices"] if all else ["-d", str(dev), "-D"])
  out = executeOutput(cmd)
  return dict((bi, int(d)) for bi, d in re.findall(_GET_BUS_INFOS, out)) \
    if all else firstGroup(_GET_BUS_INFO, out)

# - - -

if __name__ == "__main__":
  print "testing"
  _prefIdxTest()
  _time2strTest()
  _time2sTest()
  _parseWeightsTest()
  _parseIntListTest()
  _commaAndJoinTest()
  _replaceCheckTest()
  _multiMatchTest()
  _formatFloatTest()
  _repeatsTest()
  _duplicatesTest()
  _beforeAfterTest()
  _trueRegionsTest()
  _matchArraysTest()
  _turnAnglesTest()
  _invertTest()
  print "done"


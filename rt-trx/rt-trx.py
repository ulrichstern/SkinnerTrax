#
# real-time tracker
#
# 2 Jul 2015 by Ulrich Stern
#
# TODO
# * add background normalization
# * add FlyDetector stats to writeData()
# * VideoWriter:
#  - write info for dropped frames?
#  - add "no drop" mode?
#
# HOWTO
# * control LEDs based on recorded video
#  - disable "no LED control" branch in LedController's __init__
#  - reduce, e.g., postT in Protocol
#

import argparse, cv2, numpy as np, collections, array, enum, bisect
import serial, Queue, socket, threading, lockfile, random
from SocketServer import BaseRequestHandler, ThreadingTCPServer

import match_template
from util import *

# - - -

BORDER_WIDTH = 2

DEBUG = False

SHOW_EVERY_NTH = 10   # "maximum speed" for video

RESULTS_DIR = "res"
FCC = "H264" if OPCV3 else "MJPG"   # recent OpenCV 3 versions can write H.264

# - - -

def options():
  p = argparse.ArgumentParser(description='Real-time tracker.')

  g = p.add_argument_group('real-time tracker')
  g.add_argument('-v', dest='video', default=None, metavar='N',
    help='video source -- filename, device number (e.g., 0 for /dev/video0)' +
      ', or camera number (e.g., c1 for camera 1)')
  g.add_argument('--htl', dest='htl', action='store_true',
    help='high-throughput learning (HtL)')
  g.add_argument('--lgc', dest='lgc', action='store_true',
    help='large chamber (LgC)')
  g.add_argument('--yc', dest='yc', action='store_true',
    help='flies in lower half of chambers are yoked controls (for HtL and LgC)')
  g.add_argument('--fy', dest='fy', action='store_true',
    help='flip y-coordinates for lower half of chamber (for HtL)')
  g.add_argument('--iv', dest='interval', default=None, metavar='IV',
    help='track only the given interval (e.g., "01:00-01:30")')
  g.add_argument('--id', dest='initDelay', type=float,
    default=.5, metavar='F',
    help='length in seconds of init delay during which frames received ' +
      'from camera are discarded (default: %(default)s)')
  g.add_argument('--ch', dest='channel', type=int,
    default=0, metavar='N', choices=range(4),
    help='channel to use (0:B, 1:G, 2:R, 3:gray) (default: %(default)s)')
  g.add_argument('--lCh', dest='ledChannel', type=int,
    default=2, metavar='N', choices=range(3),
    help='channel to use to detect "LED on" (0:B, 1:G, 2:R) ' +
      '(default: %(default)s)')
  g.add_argument('--th', dest='bgDiffTh', type=float, default=50, metavar='F',
    help='threshold for background difference (default: %(default)s)')
  g.add_argument('--al', dest='alpha', type=float, default=.02, metavar='F',
    help='alpha for background accumulation (default: %(default)s)')
  g.add_argument('--ial', dest='initialAlpha', type=float,
    default=.1, metavar='F',
    help='initial alpha for background accumulation (default: %(default)s)')
  g.add_argument('--diffx', dest='diffx', type=float, default=.2, metavar='F',
    help='exclude pixels from background update if their background ' +
      'difference is greater than this fraction of --th (default: ' +
      '%(default)s)')
  g.add_argument('--brix', dest='brix', type=float, default=None, metavar='F',
    help='exclude pixels from background update if their brightness ' +
      'increase is larger than this fraction of --th (e.g., 0.8; ' +
      'default: %(default)s)')
  g.add_argument('--corrx', dest='corrx', type=float, default=None, metavar='F',
    help='exclude frame from background update if correlation with ' +
      'the previous frame used for background is above this threshold ' +
      '(e.g., 0.96; default: %(default)s)')
  g.add_argument('--arMin', dest='areaMin', type=float, default=40, metavar='F',
    help='exclude detected objects whose area is less than this value ' +
      '(default: %(default)s)')
  g.add_argument('--arMax', dest='areaMax', type=float,
    default=1000, metavar='F',
    help='exclude detected objects whose area is greater than this value ' +
      '(default: %(default)s)')
  g.add_argument('--thNF', dest='thNumFlies', type=int,
    default=500, metavar='N',
    help='determine number of flies in video only after this many frames ' +
      'with flies (default: %(default)s)')
  g.add_argument('--xo', dest='xo', type=int, default=0, metavar='N',
    help='x offset for HtL (default: %(default)s)')

  g = p.add_argument_group('protocol for LED control')
  g.add_argument('-f', dest='numFlies', type=int, default=2, metavar='N',
    help='start protocol only once this number of flies has been detected ' +
      'in video (default: %(default)s)')
  g.add_argument('--wN', dest='numFramesWithN', type=int,
    default=200, metavar='N',
    help='start protocol only once there were this many frames with the ' +
      'number of flies detected in video (default: %(default)s); ' +
      'threshold bypassed for HtL')
  g.add_argument('--rP', dest='reportProtocol', action='store_true',
    help='report details on protocol run')
  g.add_argument('--ps', dest='protocolStarted', action='store_true',
    help='show video only once protocol started')
  g.add_argument('--noLC', dest='noLedCtrl', action='store_true',
    help='do not control LEDs (do not connect with controller)')

  g = p.add_argument_group('video display, video writing, and other')
  g.add_argument('--shw', dest='showVideo', type=float, metavar='F',
    nargs='?', const=-1, default=None,
    help='show video at the given speed (e.g., 4 for 4x; default: ' +
      'maximum speed)')
  g.add_argument('--hm', dest='heatmap', type=float, default=5, metavar='F',
    help='use the given number of minutes for "running" heatmap ' +
      '(default: %(default)s)')
  g.add_argument('--stp', dest='stop', action='store_true',
    help='stop showing video for tracking problems and LED state changes')
  g.add_argument('--rT', dest='reportTimes', action='store_true',
    help='report times for addFrame calls')
  g.add_argument('-w', dest='writeVideo', action='store_true',
    help='write video')
  g.add_argument('--demo', dest='demo', action='store_true',
    help='write "demo" video with fly ellipses, etc. instead of plain ' +
      'video')
  g.add_argument('--wHm', dest='writeHeatmap', action='store_true',
    help='write protocol heatmap')
  g.add_argument('--ae', dest='autoExp', action='store_true',
    help='set LifeCam to auto exposure first (to get it out of "quite dark ' +
      'images" state)')

  g = p.add_argument_group('LED controller')
  g.add_argument('-s', dest='server', action='store_true',
    help='run as LED controller instead of tracker; the controller ' +
      'serves all trackers and needs to be started first')
  g.add_argument('--atp', dest='altTcpPort', action='store_true',
    help='use alternate TCP port (option for controller and tracker)')
  g.add_argument('--tlc', dest='tlcCol', metavar='N',
    nargs='?', const='0', default=None,
    help='column the LEDs are connected to (0: leftmost) on a TLC59711 ' +
      'Arduino box (default: %(const)s); can be comma-separated list')
  g.add_argument('--sp', dest='serialPort',
    default='COM9' if WINDOWS else '/dev/ttyACM0', metavar='N',
    help='name of the serial port the Arduino or Teensy is connected to ' +
      '(default: %(default)s)')
  g.add_argument('--leds', dest='setLeds', metavar='S',
    nargs='?', const='0', default=None,
    help='request controller to set all LEDs to the given values for ' +
      'the given times -- e.g., "100:2.5,0" or "100:2.5" for 100%% for ' +
      '2.5s, then 0%% (default: "%(const)s"); the full sequence can be ' +
      'repeated N times by prefixing it with "Nx"')

  return p.parse_args()

# parses the set LEDs commands, returning list with (value, time) tuples;
#  e.g., [(100., 2.5), (0, None)] for "100:2.5"
def setLedsCmds():
  def err(msg=None):
    error('cannot parse "%s"%s' %(opts.setLeds, " -- %s" %msg if msg else ""))
  mo = re.match(r'^(\d+)x(.*)$', opts.setLeds)
  cs, cmds = (mo.group(2) if mo else opts.setLeds).split(","), []
  for i, c in enumerate(cs):
    vt = c.split(":")
    try:
      v, t = float(vt[0]), None if len(vt) == 1 else float(vt[1])
    except ValueError:
      err("values and times must be float")
    if t is None and i < len(cs)-1:
      err("time required for all but last value")
    if not (0 <= v <= 100):
      err("values must be percentages")
    if t < 0:
      err("times must be non-negative")
    cmds.append((v, t))
  if mo:
    cmds = int(mo.group(1))*cmds
  if cmds[-1][1] != None:
    cmds.append((0, None))
  return cmds
 
opts = options()

# - - -

class Background:

  _USE_EVERY = 10
  _ACCUMULATE = True   # median otherwise
  _MAX_NUM_FRMS = 10   # for median
  _REPORT_DIST = False

  def __init__(self):
    self.th, self.alpha = opts.bgDiffTh, opts.initialAlpha
    self.bg = self.lstFrmAdded = self.lstFrm = None
    self.frms, self.insIdx, self.numFrms = None, None, 0   # for median
    self.tmr, self.ts = Timer(), [[], []]
    self.limBriInc, self.first = False, True
    self.skipLedOn, self.mas, self.maTh = False, [], None

  # adjust calculation -- called when protocol is started
  # TODO: should self.alpha be adjusted for HtL?
  def adjustCalculation(self):
    if opts.brix is not None:
      self.limBriInc = True
        # limit brightness increase going forward
        # note: should be called only once flies no longer in background
    if not opts.htl:
      self.alpha = opts.alpha
    self.maTh = np.mean(self.mas) + 2*np.std(self.mas)
    self.skipLedOn = True

  # returns whether to add frame to background
  def _doAdd(self, frame, idx, frameC):
    if idx % self._USE_EVERY != 0 and self.bg is not None:
      return False
    if not (self.lstFrmAdded is None or opts.corrx is None or
        normCorr(frame, self.lstFrmAdded) <= opts.corrx):
      return False
    ma = np.max(frameC[:,:,opts.ledChannel])
    if self.skipLedOn:
      return ma < self.maTh
    else:
      self.mas.append(ma)
      return True

  # slower than accumulation and not better
  # note: about 8 ms for _MAX_NUM_FRMS = 10
  def _medianBg(self, frame):
    mnf = self._MAX_NUM_FRMS
    if self.frms is None:
      self.frms = np.zeros((mnf,) + frame.shape, np.float32)
        # note: used deque before, but a fixed frms array was much faster
      self.insIdx = 0
    else:
      self.insIdx = (self.insIdx+1) % mnf
    fgm = None if self.bg is None else \
      self.bg - frame > self.th*opts.diffx
    self.frms[self.insIdx] = frame if fgm is None else \
      frame*~fgm + self.bg*fgm
    self.numFrms += 0 if self.numFrms == mnf else 1
    if self.numFrms == mnf:
      self.bg = np.partition(self.frms, mnf/2, axis=0)[mnf/2]
        # note: partition tiny bit faster than np.median(self.frms, axis=0)

  # returns np.uint8 foreground mask and whether frame was added
  def _addFrame(self, frameC, idx):
    frame = toChannel(frameC, opts.channel).astype(np.float32)
    self.lstFrm = frame
    addFrm = self._doAdd(frame, idx, frameC)
    if addFrm:
      self.lstFrmAdded = frame
      if self._ACCUMULATE:
        if self.bg is None:
          self.bg = frame
        else:
          bgMf = self.bg - frame
          msk = bgMf < self.th*opts.diffx
          if self.limBriInc:
            msk = np.logical_and(msk, bgMf > -self.th*opts.brix)
          cv2.accumulateWeighted(frame, self.bg, self.alpha,
            mask=msk.astype(np.uint8))
      else:
        self._medianBg(frame)
      if DEBUG and self.first and self.bg is not None:
        print self.bg, self.bg.shape
        self.first = False
    return None if self.bg is None else \
      (self.bg - frame > self.th).astype(np.uint8)*255, addFrm

  # returns foreground (mask) or None if not yet ready
  def addFrame(self, frame, idx):
    self.tmr.restart()
    fgm, added = self._addFrame(frame, idx)
    self.ts[added].append(self.tmr.get())
    return fgm

  # returns background or None if not yet ready
  def get(self): return astype(self.bg, np.uint8)

  # returns single-channel frame used for background difference
  def frame(self): return astype(self.lstFrm, np.uint8)

  # resets background
  def reset(self):
    self.__init__()

  # report times etc.
  def report(self):
    if self._REPORT_DIST:
      # TODO: self.ts now list of lists
      c = collections.Counter()
      for t in self.ts:
        c[round(t,3)] += 1
      print "\naddFrame time distribution:\n%s" %"\n".join(
        "  %.0f ms: %d" %(t*1000, c[t]) for t in sorted(c))
      tgt = [t for t in self.ts if t > 0.015]
      print "mean of addFrame times > 15ms: %s" %("-" if not tgt else
        "%.1f ms" %(1000*np.mean(tgt)))
    else:
      print "\nmean addFrame times:"
      for added in (False, True):
        print "  frame %sadded: %.0f ms" %(
          "" if added else "not ", 1000*np.mean(self.ts[added]))

# - - -

class FlyDetector:

  _NF_CONSECUTIVE = 1 if opts.htl else (3 if opts.lgc else 20)
  _MAX_NF_TMP_LOSS = 0
  _SORT_ELLS = False
  _FIT_RECT, _RECT_SMALLER_PRCNT = True, False
  _DEBUG = False

  def __init__(self):
    self.nchm = 20 if opts.htl else (4 if opts.lgc else 2)
    self._fgm, self.ells, self.ars = None, [], []   # last detect() call
    self.fls, self.nfLost = self.nchm*[None], self.nchm*[0]
    self.n, self.lastN, self.nfSameNCons, self.nOff = None, None, 1, False
    self.nfWithN = 0
    self.idOn, self.limC = False, None
    self.s = collections.Counter()   # integer
    self.sf = dict(arMin=1e10, arMax=0, elArTot=0)
    self.debugStp = False
    if self._DEBUG:
      print "\n>>> fly detector debug mode <<<"

  # sets the x- and y-position separators
  def setSep(self, xSep, ySep):
    assert (len(xSep)+1) * (len(ySep)+1) == self.nchm
    self.xSep, self.ySep, self.idOn = xSep, ySep, True

  # sets the chamber limits
  def setLims(self, limC, limY):
    self.limC, self.limY = limC, limY
    assert limY[0] < limY[1]

  # detects flies and fits ellipses
  # note: store all areas and learn thresholds from them?
  def detect(self, fgm):
    self._fgm = fgm
    del self.ells[:], self.ars[:]
    self.s['nf'] += 1
    cnts, hier = cv2.findContours(fgm.copy(), cv2.RETR_TREE,
      cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if hier is not None:
      sm = lg = cntLSm = 0
      for i, cnt in enumerate(cnts):
        if hier[0][i][3] != -1:   # has parent
          continue
        ar = cv2.contourArea(cnt)
        if ar < opts.areaMin:
          sm += 1
          continue
        elif ar > opts.areaMax:
          lg += 1
          continue
        if len(cnt) < 5:
          cntLSm += 1
          continue
        ell = cv2.minAreaRect(cnt) if self._FIT_RECT else cv2.fitEllipse(cnt)
        if self._RECT_SMALLER_PRCNT and self._FIT_RECT:
          self.s['rectSmllr'] += boxArea(ell) < boxArea(cv2.fitEllipse(cnt))
        self.ells.append(ell)
        self.ars.append(ar)
      if sm:
        self.s['small'] += sm
        self.s['nfSmall'] += 1
      if lg:
        self.s['nfLarge'] += 1
      if cntLSm:
        self.s['nfContour'] += 1
    if self._SORT_ELLS:
      self.ells = sorted(self.ells, key=lambda ell: ell[0][0])
      assert False   # self.ars needs to be sorted the same way
    self.check()
    self.otherStats()
    if self._DEBUG:
      self._checkObjects(cnts, hier)
    self.idFlies()

  # sets whether number of flies is off
  # note: the code below auto-detects the number of flies (self.n), which
  #  works ok for 2 flies, but so-so for 20 flies (high-throughput learning)
  def check(self):
    n = len(self.ells)
    if n:
      self.s['nfWithFlies'] += 1
      if self.s['nfWithFlies'] >= opts.thNumFlies:
        if n == self.lastN:
          self.nfSameNCons += 1
          if self.nfSameNCons > self._NF_CONSECUTIVE and n > self.n:
            self.n, self.nfWithN = n, 0
        else:
          self.lastN, self.nfSameNCons = n, 1
      if n == self.n:
        self.nfWithN += 1
    self.nOff = bool(self.n and n != self.n)
    if self.nOff:
      self.s['nfNSmall' if n < self.n else 'nfNLarge'] += 1

  # calculates additional stats
  def otherStats(self):
    if self.idOn:
      self.s['nfIdOn'] += 1
      if self.ars:
        self.sf['arMin'] = min(self.sf['arMin'], min(self.ars))
        self.sf['arMax'] = max(self.sf['arMax'], max(self.ars))
      self.sf['elArTot'] += sum(boxArea(e) for e in self.ells)
      self.s['nElls'] += len(self.ells)

  # for debugging: check detected objects
  def _checkObjects(self, cnts, hier):
    rects, ars, stp = [], [], False
    for cnt in cnts:
      ar = cv2.contourArea(cnt)
      ars.append(ar)
      if ar < opts.areaMin:
        continue
      if len(cnt) >= 5:
        if boxArea(cv2.fitEllipse(cnt)) > 3*ar:
          stp = True
      rect = cv2.minAreaRect(cnt)
      rects.append(rect)

    w, h = imgSize(self._fgm)
    stp = stp or any(not(0<=e[0][0]<w and 0<=e[0][1]<h) for e in self.ells)
    stp = len(cnts) > 5   # overrides

    if stp:
      img = self.drawEllipses()
      cv2.drawContours(img, cnts, -1, COL_B_L, -1, CV_AA)
      for rect in rects:
        cv2.ellipse(img, rect, COL_G, 1)
      cv2.imshow("debug image", img)
      print "\nnumber contours: %d, ellipses: %d" %(len(cnts), len(self.ells))
      print "areas: %s" %",".join("%.0f" %a for a in sorted(ars))
      self.debugStp = True

  # returns whether to stop for debugging and clears "stop" flag
  # note: to use debug stop: self.debugStp = True
  def debugStop(self):
    ds, self.debugStp = self.debugStp, False
    return ds

  # identifies the flies for each chamber
  def idFlies(self):
    if not self.idOn:
      return

    # pick largest object as fly for each side
    mas, nCols = self.nchm*[0], len(self.xSep)+1
    assert not self._SORT_ELLS   # for zip() below
    for (e, a) in zip(self.ells, self.ars):
      i = bisect.bisect_left(self.xSep, e[0][0]) + \
        bisect.bisect_left(self.ySep, e[0][1])*nCols
      if self.limC:
        if distance(e[0], self.limC[i][0]) > self.limC[i][1]:
          self.s['nLimC'] += 1
          continue
        if e[0][1] < self.limY[0] or e[0][1] > self.limY[1]:
          self.s['nLimY'] += 1
          continue
      if a > mas[i]:
        self.fls[i], mas[i] = e, a

    # handle lost flies
    for i, ma in enumerate(mas):
      if ma > 0:
        self.nfLost[i] = 0
      else:
        self.nfLost[i] += 1
        self.s['nLost'] += 1
        if self.nfLost[i] > self._MAX_NF_TMP_LOSS:
          self.fls[i] = None
          self.s['nLostPerm'] += 1

  # returns the number of flies that have been detected in the video so far
  def numFliesVideo(self): return self.n
  # returns the number of frames with the number of flies detected in video
  def numFramesWithN(self): return self.nfWithN

  # returns whether number of flies seems off
  def numFliesOff(self): return self.nOff

  # draws ellipses on the given image
  def drawEllipses(self, img=None):
    if img is None: img = toColor(self._fgm)
    ells = self.ellipses()
    for ell in ells:
      if ell is not None:
        cv2.ellipse(img, ell, COL_Y, 1)
    if self._DEBUG:
      txts = [str(i) if self.idOn else "%.0f" %self.ars[i]
        for i in range(len(ells))]
      for ell, txt in zip(ells, txts):
        if ell is not None:
          putText(img, txt,
            (ell[0][0] + .5*max(ell[1][0],ell[1][1]) + 3, ell[0][1]),
            (0, .5), textStyle(color=COL_W))
    return img

  # returns ellipses of objects to draw
  def ellipses(self): return self.fls if self.idOn else self.ells

  # returns areas of all detected objects
  def areas(self): return self.ars

  # returns list with the flies for each chamber (left to right, then down);
  #  each fly is Box2D tuple ((x,y), (w,h), theta) or None if tracker lost fly
  def flies(self): return self.fls
    
  def reportStats(self):
    print "\ntracking stats:"
    print "  ellipse fitting method: %s" %(
      "rectangle" if self._FIT_RECT else "ellipse")
    print "  number flies: %s" %self.n
    print "  number frames: %d" %self.s['nf']
    print "    with flies: %d" %self.s['nfWithFlies']
    print "    with too few flies: %d" %self.s['nfNSmall']
    print "    with too many flies: %d" %self.s['nfNLarge']
    print "    with small objects: %d  (number objects: %d)" %(
      self.s['nfSmall'], self.s['small'])
    print "    with large objects: %d" %self.s['nfLarge']
    print "    with contours with too few pixels: %d" %self.s['nfContour']
    nfp = self.s['nfIdOn']
    print "    with fly identification running: %s" %nfp
    if nfp > 0:
      mi, ma = self.sf['arMin'], self.sf['arMax']
      if ma > 0:
        print "      area min: %.1f, max: %.1f" %(mi, ma, )
      eat, ne = self.sf['elArTot'], self.s['nElls']
      if ne > 0:
        print "      ellipse area mean: %.1f" %(eat*np.pi/4/ne)
        if self._RECT_SMALLER_PRCNT and self._FIT_RECT:
          print "      rectangle fit smaller: {:.1%}".format(
            float(self.s['rectSmllr'])/ne)
      nl, nlp = self.s['nLost'], self.s['nLostPerm']
      print "      number of times fly lost: %d%s" %(nl,
        "" if nfp == 0 else "  ({:.2%} of time)".format(nl/(2.*nfp)))
      print "        corrected (loss for at most %s): %d" %(
        nItems(self._MAX_NF_TMP_LOSS, "frame"), nl-nlp)
      print "        not corrected: %d" %nlp
      if self.limC:
        print "      number objects outside limits: circle: %d, y: %d" %(
          self.s['nLimC'], self.s['nLimY'])

# - - -

# heatmaps of fly positions
# * "running" heatmap for last opts.heatmap minutes
# * "protocol" heatmap controlled by protocol (e.g., training time)
# TODO: start with 0s and possibly add 1 in image()
class Heatmap:

  def __init__(self):
    self._started = False

  # initializes heatmaps and starts "running" heatmap
  # div: divider to reduce frame dimensions to heatmap "bucket" dimensions
  def init(self, frm, fps, div=2):
    self.div, self.fps = float(div), fps
    self.numFrms = intR(opts.heatmap*60*fps)
    hw = intR(e/self.div for e in frm.shape[:2])
    self._map = [np.ones(hw, np.float32) for i in range(2)]
      # idx: 0: "running", 1: "protocol"
    self._cpos = collections.deque()
    self._started, self.idxErr = True, False
    self._phRunning = self._phStarted = False
    self.imgs, self.name = [], ""
    self.xySeps = None

  # resets the heatmap
  def reset(self):
    self._map[0][:,:] = 1
    self._cpos.clear()

  # sets the x- and y-position separators
  def setSep(self, xSep, ySep):
    self.xySeps = [[intR(e/self.div) for e in s] for s in (xSep, ySep)]

  # returns whether initialized
  def initialized(self): return self._started

  # starts protocol heatmap
  # note: called by each experimental fly's thread; used f == 0 instead of lock
  def start(self, f, name):
    if f == 0 and not self._phRunning:
      self.name = name
      self._phRunning = self._phStarted = True
      self._phNf = 0
      self._map[1][:,:] = 1

  # stops protocol heatmap
  def stop(self, f):
    if f == 0 and self._phRunning:
      self._phRunning = False
      self.imgs.append((self.image(True)[0], self.name))

  # returns protocol heatmap images
  def images(self): return self.imgs

  def _addSubCpos(self, cpos, add=True):
    for xy in cpos:
      try:
        yx = intR(e/self.div for e in xy[::-1])
        self._map[0][yx] += 1 if add else -1
        if self._phRunning and add:
          self._map[1][yx] += 1
      except IndexError:
        if add:
          self.idxErr = True
    if self._phRunning and add:
      self._phNf += 1

  # processes the given flies
  def process(self, fd):
    if not self._started:
      return
    cpos = tuple(e[0] for e in fd.ellipses() if e is not None)
    self._addSubCpos(cpos)
    self._cpos.append(cpos)
    if len(self._cpos) > self.numFrms:
      self._addSubCpos(self._cpos.popleft(), add=False)

  # returns heatmap image and name or (None, None) if not yet started
  # TODO: improve name
  def image(self, prtcl):
    if not self._started or (prtcl and not self._phStarted):
      return None, None
    img = heatmap(self._map[prtcl], xySeps=self.xySeps)
    nm = "%s (%.1f min)" %(self.name, self._phNf/60./self.fps) if prtcl else \
      "heatmap ({:.3g} min)".format(opts.heatmap)
    return cv2.resize(img, (0,0), fx=self.div, fy=self.div), nm

  # returns whether index error occurred during processing and clears it
  def indexError(self):
    ie, self.idxErr = self.idxErr, False
    return ie

# - - -

# LED controller
# * communicates with Arduino- or Teensy-based LED controller via USB virtual
#  serial
# * includes TCP server to allow use by multiple tracker processes
# * "LED controller" can refer to hardware or TCP server
class LedController:

  _LOG = "__ledController.log"   # not yet used

  # Arduino or Teensy
  _TYPE = "Teensy" if opts.htl else "Arduino"
  _BAUD_RATE, _ITEM_SEP, _EOM = 115200, ':', '\n'
  assert _EOM == '\n'   # for splitlines()
  _ARDUINO_RESTART_WAIT = 3   # seconds
  _QUAD_2_PIN, PIN13_LED = dict(tl=3, tr=9, bl=10, br=11), 'led'
  _QUAD_2_NUM = dict(tl=0, tr=1, bl=2, br=3)
  PULSE_WIDTH, PULSE_GAP = 16, 17

  # TCP
  _ADDRESS, _BUF_SIZE = ('localhost', 30000 + opts.altTcpPort), 1024

  def __init__(self):
    lc = "LED controller (port %d)" %self._ADDRESS[1]
    if opts.server:
      print "starting %s..." %lc
      self._initSerial()
      self._initServer()
    else:
      print "real-time tracker"
      if not opts.setLeds and (opts.noLedCtrl or not CameraControl.isCamera()):
        print "\n>>> no LED control <<<"
        self.sock = None
      else:
        self._initClient(lc)

  def _initSerial(self):
    print "\nconnecting with " + self._TYPE
    self.ser = serial.Serial(opts.serialPort, self._BAUD_RATE)
    print "waiting until %s restart done..." %self._TYPE
    time.sleep(self._ARDUINO_RESTART_WAIT)
    self.ser.flushInput()

  # for "with"
  def __enter__(self): return self
  def __exit__(self, exc_type, exc_value, traceback):
    self._cmdQ.join()
    if hasattr(self, 'ser'):
      self.ser.close()
    if hasattr(self, 's'):
      self.s.server_close()
    if hasattr(self, 'sock') and self.sock:
      self.sock.close()

  # - - - TCP server - - -

  _reqQ = Queue.Queue()   # shared between server threads

  def _initServer(self):
    print "\nnumbers of requests sent (to %s) and handled:" %self._TYPE
    self.numReq = 0
    startDaemon(self._requestHandler)
    self.s = ThreadingTCPServer(self._ADDRESS, self.RequestQueuer)
    self.s.allow_reuse_address = True
    try:
      self.s.serve_forever()
    except KeyboardInterrupt:
      print "\nstopping controller"
      pass

  class RequestQueuer(BaseRequestHandler):
    def handle(self):
      while True:
        msg = self.request.recv(LedController._BUF_SIZE)
        if not msg:
          break
        # note: after adding _cmdQ, each received msg should have only a single
        #  cmd, so the following could be reverted back to non-loop version
        msgs = msg.splitlines(True)
        if len(msgs) > 1:
          print "recv: |%s|" %msg
        for m in msgs:
          LedController._reqQ.put(m)
          self.request.sendall(m)

  def _reportNumReq(self, handled=True):
    printF('\r  %d %d' %(self.numReq, self.numReq - (0 if handled else 1)))

  def _requestHandler(self):
    while True:
      msg = self._reqQ.get()
      self.numReq += 1
      self.ser.write(msg)
      self._reportNumReq(handled=False)
      rcv = self.ser.readline()
      if int(rcv) != int(msg.split(self._ITEM_SEP)[1]):
        error("communication with %s failed" %self._TYPE)
      self._reportNumReq()

  # - - - TCP client - - -

  _cmdQ = Queue.Queue()   # shared, e.g., between _runProtocol() threads

  def _initClient(self, lc):
    self._setTlcCols()

    print "\nconnecting with %s" %lc
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.sock.connect(self._ADDRESS)
    except socket.error:
      error("cannot connect")
    startDaemon(self._commandHandler)

  def _setTlcCols(self):
    tc = opts.tlcCol
    if tc is not None:
      try:
        self.tlcCols = [int(c) for c in tc.split(',')]
      except ValueError:
        error('cannot parse column "%s"' %tc)

  def _commandHandler(self):
    while True:
      msg = self._cmdQ.get()
      self.sock.sendall(msg)
      self.sock.recv(self._BUF_SIZE)
      self._cmdQ.task_done()

  # send "led:value" command to LED controller
  def sendCmd(self, led, val):
    if not self.sock:
      return
    msg = '%s%s%d%s' %(led, self._ITEM_SEP, val, self._EOM)
    self._cmdQ.put(msg)
 
  # request LED controller to set the given LED to the given intensity
  # note: the LED identifier (led) is translated for LED controller
  #  e.g., 'tl' (quadrant) becomes pin or channel
  # side: 0: top, 1: bottom, None: both
  def setLed(self, led, percentage, side=None):
    if not self.sock:
      return
    if isinstance(led, int):
      leds = ['%d%s' %(led, '' if side is None else {0:'t', 1:'b'}[side])]
      val = round(percentage/100.*65535)
    elif opts.tlcCol is None:
      if led == self.PIN13_LED:
        leds, val = [13], 1 if percentage > 0 else 0
      else:
        leds, val = [self._QUAD_2_PIN[led]], round(percentage/100.*255)
    else:
      assert led != self.PIN13_LED
      leds = []
      for i, c in enumerate(self.tlcCols):
        if side is None or side == i:
          tlc, rgb = divmod(c, 3)
          leds.append(self._QUAD_2_NUM[led]*3 + rgb + tlc*12)
      val = round(percentage/100.*65535)
    for led in leds:
      self.sendCmd(led, val)

  # request LED controller to set all LEDs to the given intensity
  def setAll(self, percentage):
    if opts.htl:
      for f in range(40):
        self.setLed(f+00, percentage)
    else:
      for quad in self._QUAD_2_PIN:
        self.setLed(quad, percentage)

  # request LED controller to set LEDs according to command-line option
  def setLeds(self):
    cmds = setLedsCmds()
    print "\nsetting all LEDs to:"
    for v, t in cmds:
      print "  {:.3g}%{}".format(v, "" if t is None else ": %.1fs" %t)
      self.setAll(v)
      if t is not None:
        time.sleep(t)

# - - -

# protocol for turning LEDs on and off
# notes:
# * the protocol is mainly executed in two or more threads:
#  - process() is called by the "main" tracker thread for each frame
#  - _runProtocol() runs in a separate thread for each experimental fly,
#   allowing, e.g., using delays and timeouts independently
# * process(), e.g., sets self.inArea and sends events (e.g., "enter circle")
#  via self.eq
# * _sendQuitAfter() uses an additional thread to easily implement a timer
#  to end a "training" phase of the procotol
# TODO:
# * use subclasses instead of ProtocolType
#  advantage: should make individual protocols small
# * separate ProtocolManager and Protocol
class Protocol:

  _CN_2_QUAD = {1:'tl', 2:'tr', 3:'bl', 4:'br'}
  _USE_PIN13_LED = False
  MIN, HOUR = 60, 3600
  PT = enum.Enum('ProtocolType', 'circle rectangle move choice openLoop')
  PT_AREA = (PT.circle, PT.rectangle)
  _SHOW_TM, _SHOW_LIMITS = True, False

  _N_DISTS = 5   # for speed calculation

  def __init__(self, lc, fd, bg, vw, cn, hm):
    self.lc, self.fd, self.bg, self.vw, self.hm = lc, fd, bg, vw, hm
    self.quad = LedController.PIN13_LED if self._USE_PIN13_LED or cn == 0 \
      else self._CN_2_QUAD[cn%10]
    self.offset = (0 if cn%10 == 1 else 20) if opts.htl else None
    self.tmStarted = self.tmDone = False   # template match
    self.annLns, self.annLimC = [], None
    self.lPos, self.spd = None, None
    self.dists, self.dIdx = np.full(self._N_DISTS, np.nan), 0
      # TODO: by fly (see cPos)
    self.annLed = opts.htl or opts.lgc or not CameraControl.isCamera()

    # customize
    self.pt = self.PT.circle

    print "\nprotocol: %s" %self.pt.name
    if opts.htl and self.pt is self.PT.move or \
        opts.lgc and self.pt is self.PT.move:
      error('not yet implemented')   # see TODO above

    self.yco = opts.htl or opts.lgc   # yoked controls optional
    self.yc = not self.yco or opts.yc
    self.bw = BORDER_WIDTH if (opts.htl or opts.lgc) else 0
      # note: this should always be BORDER_WIDTH once matching old experiments
      #  is no longer important
    self.byFlyInit()

  # init for each experimental fly
  def byFlyInit(self):
    self.nef = nef = self.fd.nchm//(2 if self.yc else 1)
      # number of experimental flies

    self.ledVal, self.isOn = nef*[None], nef*[None]
    self.eq = [Queue.Queue() for f in range(nef)]   # event queue
    self.fns, self.info = (
      [collections.defaultdict(list) for f in range(nef)] for i in range(2))
    self.training = nef*[False]
    self.cPos, self._r = nef*[None], None   # for area (circle and rectangle)
    self.tlbr = nef*[None]                  #  protocols; cPos or tlbr non-None
    self.inArea = nef*[None]                #  iff training ongoing
    self.pos, self.yTop, self.yBottom, self.annX = (   # for choice protocol
      nef*[None] for i in range(4))

    self.lastOff, self.timeout = nef*[None], nef*[None]

    self.done = nef*[False]

    for f in range(nef):
      self._setLed(f, 0)

  def calcSpeed(self, fl1):
    pos = fl1[0] if fl1 else None
    dist = distance(self.lPos, pos) if pos and self.lPos else 0
    self.dists[self.dIdx] = dist
    self.dIdx = (self.dIdx+1) % len(self.dists)
    if not np.any(np.isnan(self.dists)):
      self.spd = np.mean(self.dists) * self.hm.fps
    if pos is not None:
      self.lPos = pos

  # processes the given flies
  def process(self, fd):
    if not self.tmDone:
      if fd.numFliesVideo() >= opts.numFlies and \
          (opts.htl or opts.lgc or fd.numFramesWithN() >= opts.numFramesWithN) \
          and not self.tmStarted:
        self.tmStarted = True
        startDaemon(self._matchTemplate)
      return
    for f, fl in enumerate(fd.flies()[:self.nef]):
      if self.pt in self.PT_AREA:
        circle = self.pt is self.PT.circle
        area = self.cPos[f] if circle else self.tlbr[f]
        if fl is not None and area:
          # inArea is None for "on border"
          if circle:
            dc = distance(area, fl[0])
            inArea = True if dc < self._r else (False if dc > self._r+self.bw
              else None)
          else:
            inArea = True if any(inRect(fl[0], ra) for ra in area) else (None
              if any(inRect(fl[0], ra, bw=self.bw) for ra in area) else False)
          if inArea is not None:
            if self.inArea[f] is not None and inArea != self.inArea[f]:
              self.eq[f].put('en' if inArea else 'ex')
            self.inArea[f] = inArea
      elif self.pt is self.PT.move:
        self.calcSpeed(fl)
        if self.spd is not None:
          self.eq[f].put('fr')
      elif self.pt is self.PT.choice:
        # previous position kept if fly lost
        if fl is not None:
          self.pos[f] = fl[0]
      elif self.pt is self.PT.openLoop:
        pass
      else:
        raise ArgumentError(self.pt)

  # called when user quits
  def quit(self):
    for f in range(self.nef):
      self._stopTraining(f)

  # notes:
  # * Standard (2-chamber) template:
  #  left chamber positions in 320x240 template:
  #   center: (45, 70)
  #   substrate borders: y = 27.5, 112.5  (approximate y-positions of LEDs)
  #   left/right edges: x = 4, 86  (chamber edges at substrate level)
  #   top/bottom edges: y = 2.5, 137
  #  x separator: 98
  # * HtL (high-throughput learning, 20-chamber) template:
  #  positions of chambers:
  #   left edge: x = 5, 149, 293, 437, 581
  #   top edge:  y = 4, 180, 355, 531
  #  chamber width/height: 80/128
  def _matchTemplate(self):
    bg, htl, lgc = self.bg.get(), opts.htl, opts.lgc
    if False:   # for debugging
      cv2.imwrite("tmp.jpg", bg)
    tm = match_template.match(
      bg if htl or lgc else cv2.resize(bg, (0,0), fx=2, fy=2),
      tmplN='HtL' if htl else ('LgC' if lgc else None))
    self._tm = np.array(tm if len(tm) > 3 else tm + [1], dtype=float)
    self._tm[:2] *= 1 if htl or lgc else .5

    self._xSep = [self._t2fX(x)
      for x in ([112, 260, 407, 554] if htl else ([268] if lgc else [98]))]
    self._ySep = [self._t2fY(y)
      for y in ([152, 332, 511] if htl else ([268] if lgc else []))]
    if not (htl or lgc):
      limC = [(self._t2f(c), self._t2fR(105)) for c in ((24, 70), (171, 70))]
      limY = [self._t2fY(y) for y in (-28, 168)]
      self.fd.setLims(limC, limY)
    self.fd.setSep(self._xSep, self._ySep)
    self.hm.setSep(self._xSep, self._ySep)
    self.nCols = len(self._xSep)+1

    if not (htl or lgc) and self._SHOW_LIMITS:
      self.annLimC = limC
      x1, x2 = (self._t2fX(x) for x in (-13, 61))
      self.annLns.extend(((x1, y), (x2, y)) for y in limY)
    if self._SHOW_TM:
      x1, x2 = (self._t2fX(x) for x in ((5, 661) if htl else
        ((4, 532) if lgc else (4, 191))))
      y1, y2 = (self._t2fY(y) for y in ((4, 658) if htl else
        ((4, 532) if lgc else (2, 137))))
      l = 40 if htl else 25
      self.annLns.extend([((x1, y1), (x1+l, y1)), ((x1, y1), (x1, y1+l)),
        ((x2, y2), (x2-l, y2)), ((x2, y2), (x2, y2-l))])

    for f in range(self.nef):
      startDaemon(self._runProtocol, (f,))

    self.bg.adjustCalculation()

    self.tmDone = True

  # turns the given template coordinates into frame coordinates
  # *note*: "standard" positions differ between regular and HtL chambers, see,
  #  e.g., _areaProtocol()
  def _t2f(self, xy, f=None):
    if (opts.htl or opts.lgc) and f is not None:
      r, c = divmod(f, self.nCols)
      if opts.htl:
        dx = 0 if c < 3 else 80-2*xy[0]   # c < 3: left edge, c >= 3: right edge
        dy = 0 if r < 2 or not opts.fy else 128-2*xy[1]
        xy = tupleAdd(xy, (144*c+5 + dx, [4, 180, 355, 531][r] + dy))
      else:
        dx = 0 if c == 0 else 244-2*xy[0]
        xy = tupleAdd(xy, (284*c+4 + dx, 284*r+4))
    return intR(tupleMul(xy, self._tm[3]) + self._tm[:2])
  def _t2fX(self, x, f=None): return self._t2f((x, 0), f)[0]
  def _t2fY(self, y, f=None): return self._t2f((0, y), f)[1]
  def _t2fR(self, r): return intR(r*self._tm[3])

  # run protocol for fly f
  def _runProtocol(self, f):
    if self.pt in self.PT_AREA:
      self._areaProtocol(f)
    elif self.pt is self.PT.move:
      assert not opts.htl   # self.spd (etc.) not by fly yet
      self._moveProtocol(f)
    elif self.pt is self.PT.choice:
      self._choiceProtocol(f)
    elif self.pt is self.PT.openLoop:
      self._openLoopProtocol(f)
    else:
      raise ArgumentError(self.pt)

    self.done[f] = True

  # - - -

  def _prePeriod(self, f, preT=None):
    self._timestamp(f, 'startPre')
    if preT:
      time.sleep(preT)

  def _startTraining(self, f, trainT, hmName=None):
    self._pulseMode()
    self._emptyEQ(f)
    self._timestamp(f, 'startTrain')
    self._sendQuitAfter(f, trainT)
    self.hm.start(f, 'training%s' %(' '+hmName if hmName else ''))
    self.lastOff[f] = time.time()
    self.training[f], self.timeout[f], self.isOn[f] = True, None, False

  def _stopTraining(self, f):
    self.training[f] = False
    self.hm.stop(f)
    self._setLed(f, 0)
    self._pulseMode(0)

  def _postPeriod(self, f, postT):
    self._timestamp(f, 'startPost')
    time.sleep(postT)

  # sets pulse mode (e.g., 5ms @ 40Hz) on Arduino (0ms: constant)
  # customize
  def _pulseMode(self, ms=0, hz=40):
    if opts.tlcCol is not None or opts.htl or opts.lgc:
      return
    self.lc.sendCmd(LedController.PULSE_WIDTH, intR(ms*10.))
    if ms != 0:
      self.lc.sendCmd(LedController.PULSE_GAP, intR(((1000./hz)-ms)*10))

  def _emptyEQ(self, f):
    while not self.eq[f].empty():
      self.eq[f].get_nowait()

  def _sendQuitAfter(self, f, secs):
    def sendQuit():
      # note: f passed via closure (same for _condPulse(), etc.)
      time.sleep(secs)
      self.eq[f].put('quit')
    startDaemon(sendQuit)
  def _waitForQuit(self, f):
    while True:
      if self.eq[f].get() == 'quit':
        break

  def _timestamp(self, f, key):
    fn = self.vw.frameNum()
    if fn is not None:
      self.fns[f][key].append(fn)

  def _addInfo(self, f, key, val):
    self.info[f][key].append(val)

  def _setLed(self, f, val, timeout=None, side=None):
    self.isOn[f] = val > 0
    if self.ledVal[f] != val:
      self.ledVal[f] = val
      self._timestamp(f, 'v'+str(val))
      def f2led(f): return f+self.offset if opts.htl else (
        self._CN_2_QUAD[f+1] if opts.lgc else self.quad)
      self.lc.setLed(f2led(f), val, side)
      if self.yco and opts.yc:
        self.lc.setLed(f2led(f+self.nef), val)
      if val == 0:
        self.lastOff[f] = time.time()
    self.timeout[f] = timeout

  def _pulse(self, f, val, secs, postSleep=0, side=None):
    self._setLed(f, val, side=side)
    time.sleep(secs)
    self._setLed(f, 0, side=side)
    time.sleep(postSleep)

  # communicates via self.lastOff, self.training, and self.done
  def _condPulse(self, f, val, secs, after):
    def condPulse():
      slpFor = 1
      while not self.done[f]:
        if self.training[f] and time.time() - self.lastOff[f] >= after:
          self._pulse(f, val, secs)
        time.sleep(slpFor)
    startDaemon(condPulse)

  # reward at a certain rate
  # notes:
  # * topIn: top (line) or in (circle), None for "position independent"
  # * reward: (ledVal, onT, interarrival time (it), it distribution)
  #  - e.g., (25, 0.25, 3, 'c')
  #  - it distribution (itDist): c=constant, e=exponential
  #  - "constant on" for onT == it and itDist == 'c'; e.g., (25, 0.1, 0.1, 'c')
  # * event-based solution possibly better for "constant on"
  def _rateReward(self, f, topIn, reward):
    if reward is None:
      return
    val, onT, it, itDist = reward
    def rateReward():
      constOn, isOn = onT == it and itDist == 'c', False
      while self.training[f] and not self.done[f]:
        on, st = None, it
        if topIn is None:
          on = True
        elif self.pos[f]:   # self.pos has previous position if fly lost
          if self.yTop[f]:
            y = self.pos[f][1]
            on = topIn and y < self.yTop[f] or not topIn and y > self.yBottom[f]
          else:
            on = topIn == (distance(self.cPos[f], self.pos[f]) < self._r)
        if on is not None:
          if constOn:
            if on != isOn:
              self._setLed(f, val if on else 0)
              isOn = on
          elif on:
            self._pulse(f, val, onT)
            st -= onT
        if itDist == 'c':
          pass
        elif itDist == 'e':
          st = np.random.exponential(st)
        else:
          raise ArgumentError(itDist)
        time.sleep(st)
      self._setLed(f, 0)
    startDaemon(rateReward)

  # open-loop reward (on/off or alternating side)
  def _openLoop(self, f, val, onT, offT=None, alt=False):
    offT = onT if offT is None else offT
    def openLoop():
      side = 0
      while self.training[f] and not self.done[f]:
        if alt:
          self._pulse(f, val, onT, side=side)
          side = 1 - side
        else:
          self._pulse(f, val, onT, postSleep=offT, side=side)
      self._setLed(f, 0)
    startDaemon(openLoop)

  # TODO: include times (nicely formatted)
  def _msg(self, f, preT, trainT, postT, msg=None):
    if f == 0:
      print "\n  training %s..." %(msg or self.pt.name)

  # - - -

  # area (circle and rectangle) protocols
  # customize
  # TODO: adjust r?
  def _areaProtocol(self, f):
    if self.pt is self.PT.circle:
      # "standard" circle positions; format: (x, y[, r])
      if opts.htl:
        cpr = dict(bottom=(21, 107),
          top=(21, 21, 21), top15=(21, 21, 15), top10=(21, 21, 10),
            top15d=(25, 25, 15), top10d=(29, 29, 10),   # difficult
          center=(40, 64, 21), center15=(40, 64, 15), center10=(40, 64, 10))
        self._r = 21
      elif opts.lgc:
        cpr = dict(side=(22, 122, 22),
          side15=(22, 122, 15), side15s=(15, 122, 15),
          side10=(22, 122, 10), side10s=(10, 122, 10),
          center=(122, 122, 22), center15=(122, 122, 15),
            center10=(122, 122, 10),
          lofc=(72, 122, 22), lofc15=(72, 122, 15), lofc10=(72, 122, 10),
          rofc=(172, 122, 22), rofc15=(172, 122, 15), rofc10=(172, 122, 10))
      else:
        cpr = dict(
          bottom=(4+22, 112.5, 22), bottom15=(4+22, 112.5, 15),
            bottom10=(4+22, 112.5, 10),
          top=(4+22, 27.5),
          center=(45, 70, 22), center15=(45, 70, 15), center10=(45, 70, 10))
        self._r = 22
    else:
      assert not opts.lgc
      if opts.htl:
        d = 0 if True else 8   # 3 vs. 4 mm
        tlbr = dict(top=(0, 0, 80, 24+d), bottom=(0, 104-d, 80, 128))
      else:
        tlbr = dict(top=(4, 2.5, 86, 27.5), bottom=(4, 112.5, 86, 137))

    # protocol type and subtype
    tp, stp = 0, 0
    if tp == 0:
      bm = 15 if opts.htl else 30
      preT, trainT, postT = bm*self.MIN, 60*self.MIN, bm*self.MIN
      if stp == 0:
        for i, pos in enumerate(("bottom", "top", "center")):
          self._areaTraining(f, preT, trainT, postT, cpr[pos], pos, i == 0)
      elif stp == 1:   # 3x
        pos = "side" if opts.lgc else "center"
        for i in range(3):
          self._areaTraining(f, preT, trainT, postT, cpr[pos],
            "%s %d" %(pos, i+1), i == 0)
      elif stp == 2:   # for LgC, etc.
        trns = [("side", "side15", "side10"),
          ("center", "center15", "center10"),
          ("lofc", "lofc15", "lofc10"),
          ("rofc", "rofc15", "rofc10"),
          ("top", "top15", "top10"),
          ("top", "top15d", "top10d")][0]
        for i, pos in enumerate(trns):
          self._areaTraining(f, preT, trainT, postT, cpr[pos], pos, i == 0)
      elif stp == 3:   # no breaks
        trns = ("bottom", "bottom15", "bottom10") if True else \
          ("center", "center15", "center10")
        for i, pos in enumerate(trns):
          self._areaTraining(f, preT, trainT,
            0 if i < len(trns)-1 else postT, cpr[pos], pos, i == 0, i == 0)
    elif tp == 1:
      preT, trainT, postT = 0, 4*self.MIN, 0
      for i in range(15):
        for pos in ("bottom", "top"):
          self._areaTraining(f, preT, trainT, postT, cpr[pos],
            "%s %d" %(pos, i+1), False, False)
    elif tp == 2:
      preT, trainT, postT = 0, 1*self.HOUR, 30*self.MIN
      pos = "bottom"
      self._condPulse(f, 25, .25, 2*self.MIN)   # match ledVal and onT
      self._areaTraining(f, preT, trainT, postT, cpr[pos], pos)
    elif tp == 3:   # rectangle
      preT, trainT, postT = 10*self.MIN, 60*self.MIN, 10*self.MIN
      rns = ('top', 'bottom')
      area = [tlbr[rn] for rn in rns] 
      for i in range(3):
        self._areaTraining(f, preT, trainT, postT, area,
          "%s %d" %(" ".join(rns), i+1), i == 0)

  # notes:
  # * preT < 0: no initial pulse
  # * area: tuple with cPos or cPos and r (circle) or list with tlbr tuples
  #  (rectangles)
  # TODO: take out 'first' argument since only used for preT?
  def _areaTraining(self, f, preT, trainT, postT, area, msg, first=False,
      pulse=True):
    self._msg(f, preT, trainT, postT, msg)

    # customize
    # note: use None for onT or offT to not time out the on or off state
    onT, offT, ledVal, prob = .25, None, 25, 1.
    offEvs = ('T',) if opts.lgc else ('T', 'ex')
    numPulses = 1   # standard: 1x 3s pulse at ledVal
    pulseVal, pulseSecs, pulseBetween, pulsePost = ledVal, 3, 3, 3

    # pre
    self._prePeriod(f)
    if first:
      time.sleep(preT)
    if pulse:
      for i in range(numPulses):
        self._pulse(f, pulseVal, pulseSecs,
          pulseBetween if i < numPulses-1 else pulsePost)

    # training
    if self.pt is self.PT.circle:
      self.cPos[f] = self._t2f(area[:2], f)
      self._addInfo(f, 'cPos', self.cPos[f])
      if len(area) > 2:
        self._r = area[2]
        self._addInfo(f, 'r', self._r)
    else:
      self.tlbr[f] = [self._t2f(ra[:2], f) + self._t2f(ra[2:], f) for
        ra in area]
      self._addInfo(f, 'tlbr', self.tlbr[f])
    self._startTraining(f, trainT, msg)
    self.timeout[f] = offT
    while True:
      try:
        ev = self.eq[f].get(timeout=self.timeout[f])
          # timeout=None: block until event (e.g., enter); no LED changes until
          #  then
      except Queue.Empty:
        ev = 'T'
      if ev == 'quit':
        break
      # events to handle: 'en', 'ex', 'T'
      inArea = self.inArea[f]
      if self.isOn[f]:
        if ev in offEvs:
          self._setLed(f, val=0, timeout=offT if inArea else None)
      else:   # off
        if ev == 'T':
          if inArea:
            self._setLed(f, val=ledVal, timeout=onT)
          elif inArea is not None:
            self.timeout[f] = None
        elif ev == 'en':
          if random.random() < prob:
            self._setLed(f, val=ledVal, timeout=onT)
    self._stopTraining(f)
    self.cPos[f] = self.inArea[f] = None

    # post
    self._postPeriod(f, postT)

  # - - -

  # customize
  # TODO: make this work for HtL, also
  def _moveProtocol(self, f):
    preT, trainT, postT = 0, 2*self.HOUR, 0*self.MIN
    ledVal = 25
    spdTh, onIfG = 2.5, False

    self._msg(f, preT, trainT, postT)

    self._prePeriod(f, preT)

    # training
    self._startTraining(f, trainT)
    while True:
      ev = self.eq[f].get()
      if ev == 'quit':
        break
      elif ev == 'fr':
        spdG = self.spd > spdTh
        on = spdG if onIfG else self.spd < spdTh
        self.onSffx = " %s %.2f" %(">" if spdG else "<", spdTh) if on else ""
        self._setLed(f, ledVal if on else 0)
    self._stopTraining(f)

    self._postPeriod(f, postT)

  # - - -

  # choice protocol
  # customize
  def _choiceProtocol(self, f):
    tp = 4   # protocol type
      # 0:top/bottom line, 1:center line, 2:circle,
      # 3:center line three different trainings,
      # 4:position independent ("non-choice or open-loop protocol")
    numTrain = 2 if tp in (0, 4) else 3

    if tp in (0, 1, 3):
      assert not opts.lgc   # lines should be adjusted
      # TODO: ct, cb should likely be shared with _areaProtocol()
      if opts.htl:
        ct, cb, d = 21, 107, 25 if tp == 0 else 43
        annX = (0, 80)
      else:
        ct, cb, d = 27.5, 112.5, 25 if tp == 0 else 42.5
        annX = (-30, 90)
      self.yTop[f], self.yBottom[f] = (self._t2fY(y, f) for y in (ct+d, cb-d))
      self.annX[f] = [self._t2fX(x, f) for x in annX]
    elif tp == 2:
      assert not (opts.htl or opts.lgc)   # circle should be adjusted
      self.cPos[f], self._r = self._t2f((45, 70)), 36

    for i in range(numTrain):
      self._choiceTraining(f, tp, i)

  # customize
  def _choiceTraining(self, f, tp, i):
    msg = str(i+1)
    if tp == 0:
      preT, trainT, postT = 30*self.MIN, 1*self.HOUR, 30*self.MIN
      reward1, reward2 = (25, 0.25, 6, 'e'), (25, 0.25, 3, 'e')
        # see _rateReward() for reward format
      if i == 0:
        rewardTB, msg = (reward1, reward2), "t:6s b:3s"
      else:
        rewardTB, msg = (reward2, reward1), "t:3s b:6s"
    elif tp in (1, 2):
      preT, trainT, postT = 5*self.MIN, 10*self.MIN, 5*self.MIN
      reward1, reward2 = (25, 0.25, 1, 'c'), (10, 0.25, 5, 'c')
      rewardTB = (reward1, reward2) if tp == 1 else (None, reward1)
        # TB or IO (in out)
    elif tp == 3:
      preT, trainT, postT = 5*self.MIN, 10*self.MIN, 30*self.MIN
      j = [0,1,2][i]
      r1 = [(25, 0.25, 1, 'c'), (25, 0.25, 1, 'c'), (25, 0.5, 5, 'c')][j]
      r2 = [(25, 0.25, 5, 'c'), (25, 0.1, 1, 'c'),  (25, 0.1, 1, 'c')][j]
      rewardTB = (r1, r2)
    elif tp == 4:
      preT, trainT, postT = 15*self.MIN, 60*self.MIN, 15*self.MIN
      rewardTB = ((25, 0.25, 3, 'c'),)   # position independent

    self._addInfo(f, 'rewardTB', rewardTB)

    self._msg(f, preT, trainT, postT, msg)

    self._prePeriod(f, preT if i == 0 else 0)

    if False:
      self._pulse(f, 25, 0.5, 3)

    self._startTraining(f, trainT, msg)
    for j, reward in enumerate(rewardTB):
      self._rateReward(f, j == 0 if len(rewardTB) > 1 else None, reward)
    self._waitForQuit(f)
    self._stopTraining(f)

    self._postPeriod(f, postT)

  # - - -

  # open-loop protocol
  # customize
  def _openLoopProtocol(self, f):
    preT, trainT, postT = 30*self.MIN, 2*self.HOUR, 30*self.MIN
    ledVal, onT, offT, self.alt = 25, 4*self.MIN, None, True

    self._msg(f, preT, trainT, postT)

    self._prePeriod(f, preT)

    # training
    self._startTraining(f, trainT)
    self._openLoop(f, ledVal, onT, offT=offT, alt=self.alt)
    self._waitForQuit(f)
    self._stopTraining(f)

    self._postPeriod(f, postT)

  # - - -

  # returns whether protocol done
  def isDone(self): return all(self.done)

  # returns dictionary with info about protocol run
  def data(self):
    def c(v):   # to make format consistent for regular chamber
      return v[0] if self.nef == 1 else v
    d = dict(frameNums=c(self.fns), info=c(self.info), pt=self.pt.name,
      ct='htl' if opts.htl else ('large' if opts.lgc else 'regular'),
      yc=self.yc, bw=self.bw, fy=opts.fy)
    if hasattr(self, '_tm'):
      d['tm'] = dict(x=self._tm[0], y=self._tm[1], fctr=self._tm[3])
    if self.cPos or self._r:
      d['circle'] = dict(pos=c(self.cPos), r=self._r)
    if self.yTop or self.yBottom:
      d['lines'] = dict(yTop=c(self.yTop), yBottom=c(self.yBottom))
    if hasattr(self, 'alt'):
      d['alt'] = self.alt
    return d

  # annotates the given image by, e.g., drawing circle
  def annotate(self, img):
    if self.annLns:
      for p1, p2 in self.annLns:
        cv2.line(img, p1, p2, COL_W)
    if self.annLimC:
      for c in self.annLimC:
        cv2.circle(img, c[0], c[1], COL_W)
    for f in range(self.nef):
      if self.training[f]:
        if self.pt in (self.PT.circle, self.PT.rectangle, self.PT.choice):
          lw = 2 if self.isOn[f] and self.annLed else 1
          if self.cPos and self.cPos[f]:
            cv2.circle(img, self.cPos[f], self._r, COL_W, lw)
          elif self.tlbr and self.tlbr[f]:
            for ra in self.tlbr[f]:
              cv2.rectangle(img, intR(ra[:2]), intR(ra[2:]), COL_W, lw)
          elif self.yTop and self.yTop[f]:
            yp, ax = None, self.annX[f]
            for y in (self.yTop[f], self.yBottom[f]):
              if y != yp:
                cv2.line(img, (ax[0], y), (ax[1], y), COL_W, lw)
                yp = y
        elif self.pt is self.PT.move:
          txt = 'speed: %s' %('-' if self.spd is None else
            '%.2f%s' %(self.spd, self.onSffx))
          putText(img, txt, (3, img.shape[0]-5), (0, 0), textStyle(color=COL_W))
    return img

# - - -

# camera control (exposure, etc.)
# notes:
# * camera control is camera specific
# * code below works for Microsoft LifeCam
# * to list available controls: v4l2-ctl -d 0 --list-ctrls-menus
# * settings are saved when the tracker exits and loaded when it starts
# * settings are saved for multiple cameras, with cameras identified by
#  their "bus info" (e.g., "usb-0000:00:14.0-1"); an alternative would
#  be to use, e.g., camera serial numbers
class CameraControl:

  # - - -

  # single V4L2 (UVC) "control" such as exposure
  class Control:

    _LIFECAM = True

    _key2ctl, _key2val = {}, {}
      # note: having _key2val static supports settings for only one camera
      #  per process

    def __init__(self, key, label, ctlName, min, max, dflt):
      self.key, self.label, self.ctlName, self.min, self.max = \
        key, label, ctlName, min, max
      self._key2ctl[key] = self
      self._key2val[key] = dflt

    # returns tuple to set control using v4l2Control()
    def setTuple(self, val=None):
      if val is None:
        val = self._key2val[self.key]
      if self is self._EXP and self._LIFECAM:
        val = [5, 10, 20, 39, 78, 156, 312, 625, 1250,
          2500, 5000, 10000, 20000][val+11]
          # converts UI (Microsoft) exposure value (-11 ... 1) to value for
          #  UVC's exposure_absolute
          # * alternative implementation: int(10000*2**val + .49)
          # * other values for exposure_absolute caused frame rate to drop
      return (self.ctlName, val)

    # sets value -- on the given device and locally
    def set(self, dev, val):
      try:
        v4l2Control(dev, self.setTuple(val))
        self._key2val[self.key] = val
      except subprocess.CalledProcessError:
        pass   # TODO: indicate V4L2 issue in UI (change color?)

    # gets value
    def get(self, k2v=None):
      return k2v[self.key] if k2v else self._key2val[self.key]

    # sets _key2val from the given dictionary, skipping deprecated keys,
    #  returning dictionary without deprecated keys
    @staticmethod
    def setValues(k2v):
      c = CameraControl.Control
      k2v = dict((k, v) for k, v in k2v.iteritems() if k in c._key2ctl)
      c._key2val.update(k2v)
      return k2v

    # reports values for all controls
    @staticmethod
    def reportValues(k2v=None):
      c = CameraControl.Control
      k2v = k2v or c._key2val
      print "  %s" %", ".join("%s:%d" %(c._key2ctl[k].label, v)
        for (k, v) in sorted(k2v.iteritems()))

  # note: use None for ctlName for "pseudo control" that is not sent to camera
  Control._EXP = Control('e', 'exp', 'exposure_absolute', -11, 1, -1)
  Control._FOC = Control('f', 'foc', 'focus_absolute', 0, 40, 20)
  Control._ZOOM = Control('z', 'zoom', 'zoom_absolute', 0, 10,
    0 if opts.htl or opts.lgc else 6)
  Control._SAT = Control('s', 'sat', 'saturation', 0, 200, 83)
  Control._CON = Control('c', 'con', 'contrast', 0, 10, 5)
  Control._BRI = Control('b', 'bri', 'brightness', 30, 255, 133)
  Control._NUM = Control('n', 'num', None, 0, 99, 0)

  # - - -

  _CTRLS_FILE = "__cameraControls"
  _TIMEOUT = 10
  _READ_NUM = 10   # for LifeCam exposure issue
  _FRAME_RATES = (7.5, 10., 15., 20., 30.)   # typical webcam frame rates
  _CAM_NUM = re.compile(r'^c(\d+)$')

  def __init__(self):
    self.isCam = self.isCamera(self)
    if self.isCam:
      self._load()
      self.ctl, self.timeout = None, None   # active control
      self.camReady = False
      assert self.dev is not None
      if self.cn is not None:
        print "device number: %d" %self.dev

  # returns whether camera, and if self given, sets self.dev and self.cn
  @staticmethod
  def isCamera(self=None):
    mo = CameraControl._CAM_NUM.match(opts.video)
    cn = int(mo.group(1)) if mo else None
    dev = None if cn else toInt(opts.video)
    if self:
      self.dev, self.cn = dev, cn
    return True if cn else isinstance(dev, int)

  # for "with"
  def __enter__(self): return self
  def __exit__(self, exc_type, exc_value, traceback):
    if self.isCam and hasattr(self.Control, '_key2val'):
      self._save()

  # sets self.bus and possibly self.dev
  def _setBus(self, bus2k2v):
    if self.dev is not None:
      try:
        self.bus = v4l2BusInfo(self.dev)
      except subprocess.CalledProcessError:
        error("cannot retrieve bus info for device %s" %self.dev +
          " (likely not connected)")
    else:
      try:
        bus2dev = v4l2BusInfo()
      except subprocess.CalledProcessError:
        error("cannot list video devices")
      cnKnown = False
      for bus, k2v in bus2k2v.iteritems():
        if self.Control._NUM.get(k2v) == self.cn:
          cnKnown = True
          if bus in bus2dev:
            dev = bus2dev[bus]
            if self.dev is not None:
              error(("two of the connected devices (%d and %d) use " +
                "camera number %d") %(self.dev, dev, self.cn))
            self.bus, self.dev = bus, dev
      if not cnKnown:
        error("camera number %d unknown" %self.cn)
      elif self.dev is None:
        error("camera %d not connected" %self.cn)

  # loads or saves the control values
  def _l_s(self, load=True):
    print "\n%s camera controls" %("loading saved" if load else "saving")
    with lockfile.LockFile(self._CTRLS_FILE):
      bus2k2v = unpickle(self._CTRLS_FILE) or {}
      if not load:
        bus2k2v[self.bus] = self.Control._key2val
        pickle(bus2k2v, self._CTRLS_FILE)
        self.Control.reportValues()
    if load:
      self._setBus(bus2k2v)
      k2v = self.Control.setValues(bus2k2v.get(self.bus, {}))
      if k2v:
        self.Control.reportValues(k2v)
      else:
        print "  no saved controls for this camera"

  def _load(self): self._l_s()
  def _save(self): self._l_s(load=False)

  # initialize camera
  def initCamera(self, cap):
    if not self.isCam:
      return
    print "initializing camera"
    # reading a certain number of frames from the VideoCapture is required
    # to wait for camera to "stabilize" under certain conditions
    def readFrames():
      print "  reading %d frames..." %self._READ_NUM
      for i in range(self._READ_NUM): cap.read()

    if False and (opts.htl or opts.lgc):
      # note: intermediate check-in; this code branch has not been properly
      #  tested (used version of cv2.so with default resolution 1280x720)
      #readFrames()
      frameSize(cap, (1280, 720))
    if self.Control._LIFECAM:
      # LifeCam exposure_auto: 1: manual, 3: auto
      if opts.autoExp:
        v4l2Control(self.dev, ("exposure_auto", 3))
        # note: if change from auto to manual exposure (code below) is done
        #  on brighter frames with _READ_NUM = 20, LifeCam can get into "quite
        #  dark images" state; using opts.autoExp gets camera out of this state
        #  without unplugging (rebooting it)
      # readFrames() required before changing from auto to manual exposure
      # to avoid "quite dark images" state for LifeCam
      if v4l2Control(self.dev, "exposure_auto") != 1:
        readFrames()

    # set camera controls
    ctls = [
      ("focus_auto", 0),
      ("exposure_auto", 1),   # see above
      ("white_balance_temperature_auto", 0),
      ("power_line_frequency", 2),   # 60 Hz
      None
    ]
    for c in self.Control._key2ctl.itervalues():
      ctls.append(c.setTuple())
    if False and opts.lgc:   # initial attempt
      ctls.extend([None, ("brightness", 79)])
    v4l2Control(self.dev, ctls, defaultOtherCtls=True)

    if self.Control._LIFECAM:
      # readFrames() required for large chamber to skip possibly unsharp frames
      # that if used for initial background caused tracking problems
      if opts.lgc:
        readFrames()

    self.camReady = True

  def _startTimeout(self):
    if self.timeout:
      self.timeout.restart()
    else:
      self.timeout = Timer()

  _UP_KEYS, _DOWN_KEYS = map(frozenset, (('+', '=', 'R'), ('-', 'T')))

  # process key press
  def processKey(self, k):
    if not (self.isCam and self.camReady):
      return
    ctl = self.ctl
    if k in self._UP_KEYS or k in self._DOWN_KEYS:
      if ctl:
        val = ctl.get() + (1 if k in self._UP_KEYS else -1)
        if ctl.min <= val and val <= ctl.max:
          ctl.set(self.dev, val)
        self._startTimeout()
    elif k in self.Control._key2ctl:
      ctl1 = self.Control._key2ctl[k]
      self.ctl = None if ctl1 == ctl else ctl1
      self._startTimeout()

  # show control on the given image
  def showControl(self, img):
    if self.isCam:
      ctl = self.ctl
      if ctl and self.timeout:
        if self.timeout.get() > self._TIMEOUT:
          self.timeout = self.ctl = None
        txt = "%s %d" %(ctl.label, ctl.get())
        tlm = np.mean(toGray(img[:10,:10,:]))
        putText(img, txt, (3,3), (0,1),
          textStyle(color=COL_W if tlm < 128 else COL_BK))
    return img

  # returns the camera number, 0 for video
  def cameraNum(self):
    return self.Control._NUM.get() if self.isCam else 0

  # returns a typical frame rate "matching" the given measured frame rate
  #  and the relative error
  @staticmethod
  def frameRate(fps):
    ds = sorted((abs(float(fps)-fr), fr) for fr in CameraControl._FRAME_RATES)
    d, fr = ds[0]
    return fr, d/fr

# - - -

# video writer
# notes:
# * use if camera's frame rate can differ from that of the video being written
#  (note: for LifeCam, the frame rate depends on the exposure)
# * drops frames if camera's frame rate exceeds video's frame rate
# * also writes trajectories (which need to be in sync with frames written)
# * stop() needs to be called at the end of video
# * the call to stop() is automatic if VideoWriter is used with "with"
class VideoWriter:

  _EXT, _TRX_FILE = ".avi", "__trx"

  # note: extension ".avi" will be added to given video filename
  def __init__(self, fn=None, fcc=FCC, fps=7.5):
    self.fn, self.dummy = fn, fn is None
    self.ts, self.x = array.array('f'), None
    if self.dummy:
      return
    print "\nwriting video to %s" %fn

    self.fcc, self.fps, self.dt = fcc, fps, 1./fps
    self.q, self._stop, self.n = Queue.Queue(), False, 0   # shared
    self.wrtr = threading.Thread(target=self._writer)
    self.wrtr.start()

  # writer thread
  def _writer(self):
    lastFrmTsFls = firstTime = vw = None
    while True:
      if self._stop:
        break
      # get most recent frame
      frmTsFls = lastFrmTsFls
      while not self.q.empty():
        frmTsFls = self.q.get_nowait()
      if frmTsFls is not None:
        lastFrmTsFls = frmTsFls
        frm, ts, fls = frmTsFls
        if vw is None:
          vw = cv2.VideoWriter(self.fn+self._EXT, cvFourcc(self.fcc),
            self.fps, imgSize(frm), isColor=numChannels(frm)>1)
          firstTime = time.time()
        vw.write(frm)
        self._appendToTrajectories(ts, fls)
        self.n += 1
      dt = self.dt if firstTime is None else max(0,
        firstTime + self.n*self.dt - time.time())
      time.sleep(dt)

  # for "with"
  def __enter__(self): return self
  def __exit__(self, exc_type, exc_value, traceback):
    if not self.dummy and not self._stop:
      self.stop()
    self._writeTrajectories()

  # append timestamp and flies to trajectories
  def _appendToTrajectories(self, ts, fls):
    if self.x is None:
      self.x, self.y, self.w, self.h, self.theta = (
        [array.array('f') for f in range(len(fls))] for i in range(5))
    self.ts.append(ts)
    nan = float('nan')
    for i, fl in enumerate(fls):
      self.x[i].append(fl[0][0] if fl else nan)
      self.y[i].append(fl[0][1] if fl else nan)
      self.w[i].append(fl[1][0] if fl else nan)
      self.h[i].append(fl[1][1] if fl else nan)
      self.theta[i].append(fl[2] if fl else nan)

  # write trajectories
  def _writeTrajectories(self):
    if self.x is not None:
      fn = self.filename()
      pickle(dict(x=self.x, y=self.y, w=self.w, h=self.h, theta=self.theta,
        ts=self.ts), fn+".trx" if fn else self._TRX_FILE)

  # write frame; can be called at rate different from fps
  def write(self, frm, ts, fd):
    fls = fd.flies()
    if not self.dummy:
      self.q.put((frm, ts, tuple(fls)))
    else:
      self._appendToTrajectories(ts, fls)

  # returns number (0,1,...) of next frame written to video; None if no video
  #  written
  def frameNum(self): return None if self.dummy else self.n

  # returns the video filename (without extension), None if no video written
  def filename(self): return self.fn

  # stop video writer
  def stop(self):
    if not self.dummy:
      self._stop = True
      self.wrtr.join()

# - - -

# video shower
# notes:
# * displays video, including background or heatmap
# * handles keyboard input
class VideoShower:

  _SHOW_AREAS = False
  _SHOW_GRAY = False   # show grayscale video that is used for tracking

  def __init__(self, winName, cc, bg, fd, hm, p):
    self.winName = winName
    self.cc, self.bg, self.fd, self.hm, self.p = cc, bg, fd, hm, p
    self.lstStp, self._quit = None, False
    self.mode = 2    # 0, 1: heatmap, 2: bg

  def _handleInput(self, delay):
    k = chr(cv2.waitKey(delay) & 0xff)
    if k == 'q':
      self._quit = True
    elif k == 'm':
      self.mode = (self.mode + 1) % 3
    elif k == 'g':
      self.bg.reset()
      self.hm.reset()
    else:
      self.cc.processKey(k)

  # show frame
  # note: annotates frame
  def show(self, frame, i, tm, fps):
    stp = opts.stop and self.fd.numFliesOff() or self.fd.debugStop()
    if stp:
      if self.lstStp == i-1:
        stp = False
      self.lstStp = i

    if self._SHOW_GRAY and not opts.writeVideo:
      frame = toColor(self.bg.frame())

    if opts.showVideo and \
        (self.p.tmDone or not opts.protocolStarted) and \
        (self.cc.isCam or i % SHOW_EVERY_NTH == 0 or stp or opts.showVideo > 0):
      xtr = ""
      if self._SHOW_AREAS:
        xtr = "  ars %s" %",".join("%.0f" %a for a in
          sorted(self.fd.areas(), reverse=True)[:4])
      self.cc.showControl(self.fd.drawEllipses(self.p.annotate(frame)))
      if self.mode != 2:
        img2 = self.hm.image(self.mode)
      else:
        img2 = (self.bg.get(), "background")
      cv2.imshow(self.winName, combineImgs([
          (frame, "frame %s%s" %(tm, xtr)), img2],
        nc=2)[0])

      delay = 1 if self.cc.isCam else (0 if stp else (
        1 if opts.showVideo < 0 else int(1000/fps/opts.showVideo)))
      self._handleInput(delay)

  # returns whether user requested quit
  def quit(self): return self._quit

# - - -

# write data; currently protocol data only
def writeData(vw, p, hm, fd):
  if opts.writeVideo:
    data = dict(command=' '.join(sys.argv), protocol=p.data(),
      flyDetector=fd)
    pickle(data, vw.filename() + ".data")
  elif opts.reportProtocol:
    print "\nprotocol data:"
    print p.data()
  if opts.writeVideo or opts.writeHeatmap:
    fn, imgs = vw.filename(), hm.images()
    if imgs:
      cv2.imwrite((fn+"__hm" if fn else "__heatmap") + ".png",
        combineImgs(imgs, nc=5)[0])

# - - -

def reportOpenCVVersion():
  print "\nOpenCV version: %s" %cv2.__version__

def reportParams():
  bdt = opts.bgDiffTh
  tfpe = "threshold for pixel exclusion"
  print "\ntracking parameters:"
  print "  channel: %s" %opts.channel
  print "  bg difference threshold: %s" %bdt
  print "  area min: %d, max: %d" %(opts.areaMin, opts.areaMax)
  print "  background calculation:"
  print "    alpha: %s, initial: %s" %(opts.alpha, opts.initialAlpha)
  print "    difference %s: %s * %s = %s" %(
    tfpe, opts.diffx, bdt, opts.diffx*bdt)
  print "    brightness increase %s: %s * %s = %s" %(
    tfpe, opts.brix, bdt, None if opts.brix is None else opts.brix*bdt)
  print "    correlation threshold for frame exclusion: %s" %opts.corrx

def track(lc, cc, vw):
  reportOpenCVVersion()
  cam = cc.isCam
  cap = videoCapture(cc.dev if cam else opts.video)
  cc.initCamera(cap)
  reportParams()
  bg, fd, hm = Background(), FlyDetector(), Heatmap()
  p = Protocol(lc, fd, bg, vw, cc.cameraNum(), hm)

  if cam:
    tfr, tfavg, i1, alp = Timer(), None, 3, .1
    i2, fps = 10, None
    winName = "camera %s" %cc.cameraNum()
  else:
    rng, fps = frameRange(cap, opts.interval), frameRate(cap)
    winName = "video %s" %basename(opts.video)
  vs = VideoShower(winName, cc, bg, fd, hm, p)

  print "\ntracking..."
  i = 0 if cam else rng[0]
  if not cam:
    setPosFrame(cap, i)
  t, frt = Timer(), ""
  while cam or i < rng[1]:
    ret, frame = cap.read()
    if not ret:
      break
    tmS = t.get()
    if cam and tmS < opts.initDelay:
      continue

    if (opts.htl or opts.lgc) and frame.shape[1] == 1280:
      xo = 280+opts.xo
      frame = frame[:,xo:xo+720,:]

    if cam:   # measure frame rate
      tf = tfr.getR()
      if i > i1:
        tfavg = tf if tfavg is None else (1-alp)*tfavg + alp*tf
        frt = " %.1f fps " %(1/tfavg)
        if fps is None and i > i2:
          fps, re = CameraControl.frameRate(1/tfavg)
          if re > 0.02:
            printF('\n  ')
            warn('measured frame rate differs from typical {:.2g} fps'.format(
              fps))
    tm = s2time(tmS) if cam else frame2time(i, fps)
    printF('\r  %s %d%s' %(tm, i, frt))

    wrtFrm = frame if opts.demo else frame.copy()

    fgm = bg.addFrame(frame, i)
    fd.detect(fgm)
    if not hm.initialized() and fps:
      hm.init(frame, fps=fps)
    hm.process(fd)
    p.process(fd)
    if p.isDone():
      break

    vs.show(frame, i, tm, fps)
    if vs.quit():
      p.quit()
      break

    vw.write(wrtFrm, tmS, fd)

    i += 1

  print "\r  done" + 30*" "

  fd.reportStats()
  print "\ntotal time: %.1fs" %t.get()
  if opts.reportTimes:
    bg.report()

  writeData(vw, p, hm, fd)
  vw.stop()
  cap.release()
  if opts.showVideo and not cam and i == rng[1]:
    cv2.waitKey(0)
  cv2.destroyAllWindows()

def main():
  with LedController() as lc:
    if not opts.server:
      if opts.setLeds:
        lc.setLeds()
        return
      with CameraControl() as cc:
        fn = None
        if opts.writeVideo:
          if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
          fn = os.path.join(RESULTS_DIR, "c%s__%s" %(cc.cameraNum(),
            time2str(format='%Y-%m-%d__%H-%M-%S')))
        with VideoWriter(fn) as vw:
          track(lc, cc, vw)

# - - -

main()


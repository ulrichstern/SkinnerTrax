# -*- coding: utf-8 -*-
#
# analyze learning experiments
#
# 18 Sep 2015 by Ulrich Stern
#
# notes:
# * naming:
#  calculated reward: entering of actual or virtual (fly 2) training circle
#  control reward: entering of control circle ("top vs. bottom")
#
# TODO
# * always for new analysis: make sure bad trajectory data skipped
#  - check this for recent additions
#  - if checkValues() is used, this is checked
# * rewrite to store data for postAnalyze() and writeStats() in dict?
# * rename reward -> response (where better)
# * write figures
# * compare tracking with Ctrax?
# * separate options for RDP and epsilon
# * fly 0, 1, and 2 used in comments
# * move CT to common.py?
#

from __future__ import division

import argparse, cv2, numpy as np, scipy.stats as st, scipy.io as sio
import scipy.ndimage as ndi
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
import collections, random, enum, textwrap
import shapely.geometry as sg, shapely.affinity as sa, pylru

from util import *
from common import *

CAM_NUM = re.compile(r'^c(\d+)__')
LOG_FILE = "__analyze.log"
STATS_FILE, VIDEO_COL = "learning_stats.csv", True
ANALYSIS_IMG_FILE = "imgs/analysis.png"
CALC_REWARDS_IMG_FILE = "imgs/%s_rewards_fly_%d.png"
REWARD_PI_IMG_FILE = "imgs/reward_pi__%s_min_buckets.png"
REWARD_PI_POST_IMG_FILE = "imgs/reward_pi_post__%s_min_buckets.png"
REWARDS_IMG_FILE = "imgs/rewards__%s_min_buckets.png"
DELAY_IMG_FILE = "imgs/delay.png"
TRX_IMG_FILE, TRX_IMG_FILE2 = "imgs/%s__t%d.png", "imgs/%s__t%d_b%d%s.png"
RUN_LENGTHS_IMG_FILE = "imgs/run_lengths.png"
TURN_ANGLES_IMG_FILE = "imgs/turn_angles.png"
HEATMAPS_IMG_FILE = "imgs/heatmaps%s.png"
OPEN_LOOP_IMG_FILE = "imgs/open_loop.png"

P = False     # whether to use paper style for plots
F2T = True    # whether to show only first two trainings for paper
LEG = False   # whether to show legend for paper

BORDER_WIDTH = 1
RDP_MIN_LINES = RDP_MIN_TURNS = 100   # for including fly in analysis
_RDP_PKG = False

SYNC_CTRL = False        # whether to start sync buckets after control reward
ST = enum.Enum('SyncType', 'fixed midline control')
  # fixed: not behavior dependent
  # midline defaults to control if training has no symmetric control circle
POST_SYNC = ST.fixed         # when to start post buckets
RI_START = ST.midline        # when to start RI calculation
RI_START_POST = ST.control   # ditto for post period

HEATMAP_DIV = 2
BACKGROUND_CHANNEL = 0   # blue (default for tracking)
SPEED_ON_BOTTOM = True   # whether to measure speed only on bottom
LEGACY_YC_CIRCLES = False   # whether to use calculated template match values
                            #  for yoked control circles

POST_TIME_MIN = False

OP_LIN, OP_LOG = 'I', 'O'
OPTS_HM = (OP_LIN, OP_LOG)

# - - -

class FlyDetector: pass   # required for unpickle()

# - - -

def options():
  p = argparse.ArgumentParser(description='Analyze learning experiments.')

  p.add_argument('-v', dest='video', default=None, metavar='N',
    help='video filename, directory name, or comma-separated list of names ' +
      '(names can contain wildcards *, ?, and []); use | to separate ' +
      'video groups (for rewards plots, etc.); use : to give fly number ' +
      'range (overriding -f for the video)')
  p.add_argument('-f', dest='fly', default=None, metavar='N',
    help='fly numbers in case of HtL or large chamber (e.g., "0-19" or ' +
      '"6-8,11-13"); use | for by-group fly numbers')
  p.add_argument('--gl', dest='groupLabels', default=None, metavar='N',
    help='labels for video groups (bar-separated)')
  p.add_argument('--aem', dest='allowMismatch', action='store_true',
    help='allow experiment descriptor mismatch, which leads to error otherwise')

  g = p.add_argument_group('specialized analysis')
  g.add_argument('--move', dest='move', action='store_true',
    help='analyze "move" experiments (not auto-recognized)')
  g.add_argument('--ol', dest='ol', action='store_true',
    help='analyze "open loop" experiments; not needed for on-off and ' +
      'alternating side protocols')
  g.add_argument('--thm', dest='thm', action='store_true',
    help='analyze trajectory heatmaps (see also --pltThm)')
  g.add_argument('--rdp', dest='rdp', type=float, metavar='F',
    nargs='?', const=3., default=0,
    help='analyze trajectories simplified using RDP with the given epsilon ' +
      '(default: %(const)s)')

  g = p.add_argument_group('tweaking analysis')
  g.add_argument('--shBB', dest='showByBucket', action='store_true',
    help='show rewards by "bucket" (--nb per training)')
  g.add_argument('--nb', dest='numBuckets', type=int,
    default=None, metavar='N',
    help='number of buckets per training (default: 1 if choice else 12)')
  g.add_argument('--nrc', dest='numRewardsCompare', type=int,
    default=100, metavar='N',
    help='number of rewards to compare (default: %(default)s)')
  g.add_argument('--sb', dest='syncBucketLenMin', type=float,
    default=10, metavar='F',
    help='length of sync buckets (in minutes, default: %(default)s); ' +
      'synchronized with first reward')
  g.add_argument('--piTh', dest='piTh', type=int,
    default=10, metavar='N',
    help='calculate reward PI only if sum is at least this number ' +
      '(default: %(default)s)')
  g.add_argument('--adbTh', dest='adbTh', type=int, default=5, metavar='N',
    help='calculate average distance traveled (or maximum distance reached) ' +
      'between rewards for sync buckets only ' +
      'if number of rewards is at least this number (default: %(default)s)')
  g.add_argument('--pib', dest='piBucketLenMin', type=float,
    default=None, metavar='F',
    help='length of post training buckets for positional PI (in minutes, ' +
      'default: 10 if choice else 2)')
  g.add_argument('--rm', dest='radiusMult', type=float,
    default=1.3, metavar='F',
    help='multiplier for radius for positional PI (default: %(default)s)')
  g.add_argument('--pb', dest='postBucketLenMin', type=float,
    default=3, metavar='F',
    help='length of post training buckets for number rewards (in minutes, ' +
      'default: %(default)s)')
  g.add_argument('--rpib', dest='rpiPostBucketLenMin', type=float,
    default=3, metavar='F',
    help='length of post training buckets for reward PI (in minutes, ' +
      'default: %(default)s)')
  g.add_argument('--skp', dest='skip', type=float,
    default=0, metavar='F',
    help='skip the given number of minutes from beginning of buckets ' +
      '(default: %(default)s)')
  g.add_argument('--skpPi', dest='skipPI', action='store_true',
    help='if fly did not visit both top and bottom during bucket\'s ' +
      '--skp period, skip bucket\'s PI in %s' %STATS_FILE)
  g.add_argument('--minVis', dest='minVis', type=int,
    default=0, metavar='N',
    help='skip bucket\'s PI in %s unless each top and bottom ' %STATS_FILE +
      'were visited at least this many times (default: %(default)s)')

  g = p.add_argument_group('plotting')
  g.add_argument('--shPlt', dest='showPlots', action='store_true',
    help='show plots')
  g.add_argument('--fs', dest='fontSize', type=float,
    default=mpl.rcParams['font.size'], metavar='F',
    help='font size for plots (default: %(default)s)')
  g.add_argument('--ws', dest='wspace', type=float,
    default=mpl.rcParams['figure.subplot.wspace'], metavar='F',
    help='width of space between subplots (default: %(default)s)')
  g.add_argument('--pltAll', dest='plotAll', action='store_true',
    help='plot all rewards')
  g.add_argument('--pltTrx', dest='plotTrx', action='store_true',
    help='plot trajectories (plot depends on protocol)')
  g.add_argument('--pltThm', dest='plotThm', action='store_true',
    help='plot trajectory heatmaps')
  g.add_argument('--pltHm', dest='hm', choices=OPTS_HM,
    nargs='?', const=OP_LOG, default=None,
    help='plot heatmaps with linear (%s) or logarithmic (%s, default) colorbar'
      %OPTS_HM)
  g.add_argument('--bg', dest='bg', type=float,
    nargs='?', const=.6, default=None, metavar='F',
    help='plot heatmaps on chamber background with the given alpha ' +
      '(default: %(const)s); use 0 to show chamber background')
  g.add_argument('--grn', dest='green', action='store_true',
    help='use green for LED color')
  g.add_argument('--fix', dest='fixSeed', action='store_true',
    help='fix random seed for rewards images')

  g = p.add_argument_group('rt-trx "debug"')
  g.add_argument('--shTI', dest='showTrackIssues', action='store_true',
    help='show tracking issues')
  g.add_argument('--shRM', dest='showRewardMismatch', action='store_true',
    help='show mismatch between calculated and actual rewards ' +
      '(typically due to dropped frames in rt-trx.py)')
  g.add_argument('--dlyCk', dest='delayCheckMult', type=float, metavar='F',
    nargs='?', const=3, default=None,
    help='check delay between response and "LED on," using the given ' +
      'standard deviation multiplier to set the "LED on" threshold ' +
      '(default: %(const)s)')

  g = p.add_argument_group('specialized files and player')
  g.add_argument('--ann', dest='annotate', action='store_true',
    help='write annotated video')
  g.add_argument('--mat', dest='matFile', action='store_true',
    help='write MATLAB file (see yanglab Wiki for fields)')
  g.add_argument('--play', dest='play', action='store_true',
    help='play annotated video')
  return p.parse_args()

# set option defaults depending on protocol
def setOptionDefaults(va):
  if hasattr(opts, '_dfltsSet'):
    return
  opts._dfltsSet = True
  if opts.numBuckets is None:
    opts.numBuckets = 1 if va.choice else 12
  if opts.piBucketLenMin is None:
    opts.piBucketLenMin = 10 if va.choice else 2

def skipMsg():
  return "(first %s min of each bucket skipped)" %formatFloat(opts.skip, 1)

opts = options()

# - - -

def frame2hm(nf, fps):
  h = nf/fps/3600
  return "%.1fh" %h if h >= 1 else "%s min" %formatFloat(h*60, 1)

def cVsA(calc, ctrl=False, abb=True):
  return ("ctrl." if abb else "__control__") if ctrl else (
    ("calc." if abb else "__calculated__") if calc else "actual")
def cVsA_l(calc, ctrl=False): return cVsA(calc, ctrl, False)

# capitalize for paper
def pcap(s): return s[:1].upper() + s[1:] if P else s

def pch(a, b): return a if P else b

# - - -

# minimal wrapper for training
# notes:
# * data attributes (e.g., start, stop, etc.) are accessed w/out method
# * naming virtual vs. control circles, see comment at beginning of file
class Training:

  TP = enum.Enum('TrainingType', 'bottom top center circle choice move')
    # circle is used for various positions in large chamber
  HAS_SYM_CTRL = {TP.bottom, TP.top}

  _exp, _expVals = None, None

  # n = 1, 2, ...
  def __init__(self, n, start, stop, va, circle=None, ytb=None):
    self.n, self.start, self.stop, self.va = n, start, stop, va
    self.ct, self.xf, self.fps, self.yc = va.ct, va.xf, va.fps, not va.noyc
    (self.cx, self.cy), self.r = circle if circle else ((None, None), None)
    (self.yTop, self.yBottom) = ytb if ytb else (None, None)
    self.cs, self.v_cs = [], []   # training and control circles for each fly
    self._setCntr()
    self.sym = False

  def _setCntr(self):
    if not hasattr(self, 'cntr') and self.xf.initialized():
      self.cntr = self.xf.t2f(*self.ct.center(), f=self.va.ef)

  def isCircle(self): return self.cx is not None

  # length in frames
  def len(self, post=False):
    return self.postStop - self.stop if post else self.stop - self.start

  # returns whether this training has symmetrical control circle
  def hasSymCtrl(self): return self.tp in self.HAS_SYM_CTRL or self.sym

  # returns training and control circle(s) for the given fly
  def circles(self, f=0):
    return self.v_cs if f == 1 else self.cs

  # returns name (short version: e.g., "training 1")
  def name(self, short=True):
    if not short:
      tt, pt = (frame2hm(self.len(x), self.fps) for x in (False, True))
    return "%s %d%s" %(pch("session", "training"), self.n,
      "" if short else ": %s, %s (post: %s)" %(tt, self.tpS, pt))
  # returns short name (e.g., "t1")
  def sname(self): return "t%d" %self.n

  # draws, e.g., circles on the given image
  # ctrl: False: exp. circle, True: control circle, None: all circles
  # returns cx, cy, and r in case of single circle
  def annotate(self, img, ctrl=False, col=COL_W, f=0):
    if self.cs:
      cs = self.cs + self.v_cs if ctrl is None else \
        self.circles(f)[ctrl:ctrl+1]
      for cx, cy, r in cs:
        cv2.circle(img, (cx, cy), r, col)
      if len(cs) == 1:
        return cs[0]
    elif self.tp is self.TP.choice:
      for y in (self.yTop, self.yBottom):
        (xm, ym), (xM, yM) = self.ct.floor(self.xf, f=self.va.ef)
        bw = {CT.regular: -12, CT.htl: 15, CT.large: 35}[self.ct]
        cv2.line(img, (xm-bw, y), (xM+bw, y), col)

  # returns the training for the given frame index, None for non-training
  @staticmethod
  def get(trns, fi, includePost=False):
    for t in trns:
      if t.start <= fi < (t.postStop if includePost else t.stop):
        return t
    return None

  # sets training, control, and virtual (yoked control) circles
  @staticmethod
  def _setCircles(trns, cyu):
    if not any(t.isCircle() for t in trns):
      return
    calcTm, xf = len(cyu) == 3, trns[0].xf
    if calcTm and trns[0].ct is CT.regular:   # calculate template match values
      tmFct = (cyu[2]-cyu[0])/(112.5-27.5)
      xm, ym = [min(t.cx if i else t.cy for t in trns if t.isCircle())
        for i in (1, 0)]
      tmX, tmY = xm - (4+22)*tmFct, ym - 27.5*tmFct
      if not xf.initialized():
        xf.init(dict(fctr=tmFct, x=tmX, y=tmY))
        for t in trns: t._setCntr()
      else:
        errs = abs(xf.x-tmX), abs(xf.y-tmY), abs(xf.fctr-tmFct)/tmFct
        assert all(err < .7 for err in errs[:2]) and errs[2] < .01
    else:
      tmFct, tmX = xf.fctr, xf.x
    for t in trns:
      if t.isCircle():
        isCntr = t.tp is t.TP.center
        def addC(cx, cy, r):
          t.cs.append((cx, cy, r))
          if t.ct is CT.regular:   # for yoked control
            assert t.yc
            ccx = 150.5 if isCntr else 192-22
            ccx = intR(ccx*tmFct + tmX) if LEGACY_YC_CIRCLES else xf.t2fX(ccx)
            t.v_cs.append((ccx, cy, r))
          elif t.yc and t.ct is CT.large:
            t.v_cs.append((cx, 2*xf.t2fY(268) - cy, r))
        addC(t.cx, t.cy, t.r)
        # add control circle
        if t.tp is t.TP.circle:
          if t.ct is CT.large:
            addC(t.cx, t.cy, 55)   # 22*3 scaled for large chamber
          elif t.ct is CT.htl:
            addC(t.cx, 2*t.cntr[1]-t.cy, t.r)
            t.sym = True
          else:
            error('TrainingType circle not implemented for %s chamber' %t.ct)
        elif isCntr:
          assert len(cyu) != 3 or t.cy == cyu[1]
          addC(t.cx, t.cy, intR(t.r*(2.5 if t.ct is CT.htl else 3)))
        else:
          if len(cyu) == 3:
            assert t.cy == cyu[0] or t.cy == cyu[2]
            ccy = cyu[2] if t.cy == cyu[0] else cyu[0]
          elif t.tp in (t.TP.bottom, t.TP.top):
            assert t.ct is CT.regular
            ccy = xf.t2fY(112.5 if t.tp is t.TP.top else 27.5)
          else:
            error('not yet implemented')
          addC(t.cx, ccy, t.r)

  @staticmethod
  def _setYTopBottom(trns):
    for t in trns:
      if t.tp is t.TP.choice and t.yTop is None:
        t.yTop = t.yBottom = t.xf.t2fY(t.ct.center()[1], f=t.va.ef)

  # to catch cases where the different videos (experiments) do not match
  # descriptor examples:
  # * bottom 1.0h, top 1.0h, center 1.0h
  # * 10 min x3
  @staticmethod
  def _setExperimentDescriptor(trns):
    if trns[0].isCircle():
      exp = ", ".join("%s %s" %(t.tpS, frame2hm(t.len(), t.fps)) for t in trns)
    else:
      tms = repeats([frame2hm(t.len(), t.fps) for t in trns])
      exp = ", ".join("%s%s" %(t, " x%d" %r if r > 1 else "") for (t, r) in tms)
    expVals = concat(t.expVals for t in trns)
    if Training._exp is None:
      Training._exp, Training._expVals = exp, expVals
    else:
      em = exp == Training._exp
      evm = isClose(expVals, Training._expVals, atol=1)
      if not (em and evm) and not opts.annotate and not opts.rdp and \
          not opts.allowMismatch:
        error('\nexperiment%s do not match (%s vs. %s)' %(
          ("s", '"%s"' %exp, '"%s"' %Training._exp) if not em else
          (" values", "[%s]" %join(", ", expVals, p=0),
            "[%s]" %join(", ", Training._expVals, p=0))))

  # post stops on possible wake-up pulse
  @staticmethod
  def _setPostStop(trns, on, nf):
    for i, t in enumerate(trns):
      t.postStop = trns[i+1].start if i+1 < len(trns) else nf
      on = on[(t.stop < on) & (on < t.postStop)]
      if len(on):
        t.postStop = on[0]
      if POST_TIME_MIN and not opts.move and t.postStop - t.stop < 10*t.fps:
        error('less than 10s post time for %s' %t.name())

  # processes all trainings and reports trainings
  # note: call before calling instance methods
  @staticmethod
  def processReport(trns, on, nf):
    assert all(t.n == i+1 for i, t in enumerate(trns))
    Training._setPostStop(trns, on, nf)
    cyu, cxu = np.unique([t.cy for t in trns]), np.unique([t.cx for t in trns])
    # set training type
    for t in trns:
      if opts.move:
        t.tp = t.TP.move
      elif t.cx is None:
        t.tp = t.TP.choice
      else:
        cir = t.tp = "circle x=%d,y=%d,r=%d" %(t.cx, t.cy, t.r)
      if t.isCircle():
        if t.ct is CT.large:
          t.tp = t.TP.circle
        elif len(cyu) == 3 and len(cxu) == 2:
          if t.cy == cyu[2]: t.tp = t.TP.bottom
          elif t.cy == cyu[0]: t.tp = t.TP.top
          else: t.tp = t.TP.center
        else:
          def equal1(tp1, tp2):   # possibly move to util.py
            return all(abs(e1-e2) <= 1 for e1, e2 in zip(tp1, tp2))
          cc = (t.cx, t.cy)
          if t.ct is CT.htl:
            if equal1(cc, t.cntr):
              t.tp = t.TP.center
            else:
              t.tp = t.TP.circle
          else:
            assert t.ct is CT.regular
            if equal1(cc, t.cntr):
              t.tp = t.TP.center
            elif equal1(cc, t.xf.t2f(26, 112.5)):
              t.tp = t.TP.bottom
            elif equal1(cc, t.xf.t2f(26, 27.5)):
              t.tp = t.TP.top
            else:
              error('not yet implemented')
      t.expVals = t.xf.f2t(t.cx, t.cy, f=t.va.ef) + (t.r,) \
        if t.tp is t.TP.circle else ()
      t.tpS = t.tp if isinstance(t.tp, str) else t.tp.name
      print "  %s%s" %(t.name(short=False),
        " (%s)" %cir if t.isCircle() else "")
    Training._setCircles(trns, cyu)
    Training._setYTopBottom(trns)
    Training._setExperimentDescriptor(trns)

# - - -

# trajectory of one fly
class Trajectory:

  JMP_LEN_TH, DIST_TH = 30, 10
  SUSP_FRAC_TH, SUSP_NUM_TH = .03, 3
  VEL_A_MIN_D = 3
  _DEBUG = False

  # f: 0: experimental fly ("fly 1"), 1: yoked control
  def __init__(self, xy, wht=None, f=0, va=None, ts=None):
    self.x, self.y = xy
    (self.w, self.h, self.theta) = wht if wht else 3*(None,)
    self.f, self.va, self.ts = f, va, ts
    self._p("fly %d" %(f+1))
    if self._isEmpty():
      return
    self._interpolate()
    self._calcDistances()
    if self.va:
      self._calcSpeeds()
      self._calcAreas()
      self._setWalking()
      self._setOnBottom()
    self._calcAngles()
    self._suspiciousJumps()
    self._calcRewards()
    if opts.showTrackIssues:
      self._plotIssues()

  def _p(self, s):
    if self.va:
      print s

  def _isEmpty(self):
    if np.count_nonzero(np.isnan(self.x)) > .99*len(self.x):
      self._p("  no trajectory")
      self._bad = True
      return True
    return False

  def _interpolate(self):
    self.nan = np.isnan(self.x)
    self.nanrs = nanrs = trueRegions(self.nan)
    if len(nanrs) and nanrs[0].start == 0:
      del nanrs[0]
    ls = [r.stop-r.start for r in nanrs]
    self._p("  lost: number frames: %d (%s)%s" %(sum(ls),
      "{:.2%}".format(len(ls)/len(self.x)),
      "" if not ls else ", sequence length: avg: %.1f, max: %d" %(
        sum(ls)/len(ls), max(ls))))

    # lost during "on"
    if self.va:
      msk, nfon = np.zeros_like(self.x, bool), 2
      for d in range(nfon):
        msk[self.va.on+1+d] = True
      nf, nl = np.sum(msk), np.sum(msk & self.nan)
      if nf:
        print '    during "on" (%d frames, %d per "on" cmd): %d (%s)' %(
          nf, nfon, nl, "{:.2%}".format(nl/nf))

    self._p("    interpolating...")
    for r in nanrs:
      f, t = r.start, r.stop
      assert f > 0
      for a in (self.x, self.y):
        a[r] = np.interp(range(f, t),
          [f-1, t], [a[f-1], a[t] if t < len(a) else a[f-1]])

  # note: self.d is forward-looking (i.e., between current and next position),
  #  self.dBw is backward-looking
  def _calcDistances(self):
    self.d = np.full_like(self.x, np.nan)
    self.d[:-1] = distances((self.x, self.y))
    self.mean_d, self.std_d = np.nanmean(self.d), np.nanstd(self.d)
    self.d[np.isnan(self.d)] = 0
    self.dBw = np.zeros_like(self.x)
    self.dBw[1:] = self.d[:-1]

  # note: backward-looking
  def _calcSpeeds(self):
    self.sp = self.dBw * self.va.fps

  # note: of ellipse; not interpolated
  def _calcAreas(self):
    self.ar = self.w*self.h*np.pi/4

  def _setWalking(self):
    self.pxPerMmFloor = self.va.ct.pxPerMmFloor()
    self.walking = self.sp > 2*self.pxPerMmFloor   # 2 mm/s * px_per_mm
    # note: could write code to automaticalliy fix c4__2015-09-16__10-15-38.avi
    #  problem (fly resting and tracking problem makes it look like back and
    #  forth jumps).  E.g., resting, movement, flag all immediately following
    #  (resting allowed) movements that match or reverse match this movement
    #  (and the movement itself)

  # notes:
  # * False if data missing
  # * constants for both versions of "on bottom" calculation determined using
  #  _playAnnotatedVideo(), see yanglab Wiki
  # * onBottomPre: current and preceding frames are "on bottom" ("matching"
  #  self.sp)
  # TODO
  # * exclude sidewall for HtL chamber
  # * rename onBottom -> onFloor
  def _setOnBottom(self):
    if self.va.ct is CT.regular:
      v = 2   # version (1 or 2)
      xf, dx, useMinMax = self.va.xf, 15, True
      xm, ym = xf.t2f((4, 109+dx)[self.f], 2.5)
      xM, yM = xf.t2f((86-dx, 191)[self.f], 137)
      xmin, xmax = np.nanmin(self.x), np.nanmax(self.x)
      if useMinMax:
        xm = xmin+dx if self.f == 1 else xm
        xM = xmax-dx if self.f == 0 else xM
      with np.errstate(invalid='ignore'):   # suppress warnings due to NaNs
        onB = (xm < self.x) & (self.x < xM) & (ym < self.y) & (self.y < yM) & \
          (self.ar < (300 if v == 1 else 310))
      if v == 2:
        onB &= self.d < 30   # exclude jumps
        for s in trueRegions(onB):
          ar = self.ar[s.start:s.stop]
          mar = np.mean(ar)
          if mar < 210 or mar < 240 and len(ar) > 2:
            idxs = np.flatnonzero(ar < 260)   # exclude large start or stop
            onB[s.start:s.start+idxs[0]] = False
            onB[s.start+idxs[-1]+1:s.stop] = False
            continue
          onB[s.start:s.stop] = False
    elif self.va.ct is CT.htl:
      onB = ~self.nan
    elif self.va.ct is CT.large:
      onB = ~self.nan
    else:
      error('not yet implemented')
    self.onBottom = onB
    self.onBottomPre = np.zeros_like(self.x, dtype=bool)
    self.onBottomPre[1:] = self.onBottom[:-1]
    self.onBottomPre &= self.onBottom
    assert np.count_nonzero(self.onBottom != self.onBottomPre) == len(
      trueRegions(self.onBottom))
    if self.va.ct is CT.regular:
      self.dltX = np.abs(self.x - xf.t2fX((86, 109)[self.f]))
      self.dltX2 = np.abs(self.x - (xmin if self.f else xmax))

  def _calcAngles(self):
    return
      # note: to avoid strong noise effects, the calculation below needs to
      #  be improved or trajectories need to be smoothed first
    self.velA = velocityAngles(self.x, self.y)
    self.velA[self.d<self.VEL_A_MIN_D] = np.nan
    self.velAD = np.mod(np.diff(self.velA)+np.pi, 2*np.pi) - np.pi
      # TODO: should be renamed turn angle

  # check for suspicious jumps
  # note: mean_d and std_d could be used instead of constant thresholds
  def _suspiciousJumps(self):
    self.susp = []
    jis = (self.d > self.JMP_LEN_TH).nonzero()[0]
      # indexes of jump start points; jis+1 gives jump end points
    ns, nj = 0, len(jis)
    for i, ji in enumerate(jis):
      if i > 0:
        pji = jis[i-1]
        if self._DEBUG and i < 10 and self.f == 1:
          print i, ji, self.d[ji-2:ji+2]
        if self.dist(pji+1, ji) + self.dist(pji, ji+1) < self.DIST_TH:
          self.susp.extend((pji, ji))
          ns += 1
    sf = ns/nj if nj else 0
    self._bad = sf >= self.SUSP_FRAC_TH and ns >= self.SUSP_NUM_TH
    self._p("  long (>%d) jumps: %d, suspicious: %d%s%s" %(self.JMP_LEN_TH,
      nj, ns, " ({:.1%})".format(sf) if nj else "",
      " *** bad ***" if self._bad else ""))

  # compare calculated rewards with actual ones
  # note: rt-trx.py's VideoWriter can drop frames, which can result in
  #  actual rewards without calculated ones
  def _checkRewards(self, t, en):
    if self.f != 0:   # only for fly 1
      return
    en = inRange(en, t.start, t.stop)
    on = self.va._getOn(t)
    if np.array_equal(en, on):
      return
    enS, onS = set(en), set(on)
    sd = np.array(sorted(enS ^ onS))

    # skip 1-frame differences
    d1 = (np.diff(sd) == 1).nonzero()[0]
    sdS = set(np.delete(sd, np.concatenate((d1, d1+1))))
    # skip last training frame
    sdS -= {t.stop-1}

    self.no_en += len(sdS & enS)
    self.no_on += len(sdS & onS)
    if opts.showRewardMismatch:
      imgs, hdrs, nr = [], [], 4
      for j, fi in enumerate(sorted(sdS)):
        i1, i2 = fi-2, fi+3
        imgs.extend(self._annImgs(i1, i2, show='d'))
        for i in range(i1, i2):
          if i == fi:
            hdr = "f %d only %s" %(i, cVsA(fi in enS))
          else:
            hdr = "f %+d" %(i-fi)
            if i == i1 and j%nr == 0:
              hdr += "  (t %d-%d)" %(t.start, t.stop)
          hdrs.append(hdr)
        if (j+1) % nr == 0 or j+1 == len(sdS):
          self.rmImNum += 1
          cv2.imshow("reward mismatch %d" %self.rmImNum,
            combineImgs(imgs, hdrs=hdrs, nc=i2-i1)[0])
          del imgs[:], hdrs[:]

  def calcRewardsImg(self):
    for ctrl in (False, True):
        # post rewards shown for ctrl == False
      imgs, hdrs = [], []
      for t in self.va.trns:
        en = self.en[ctrl]
        fi, la = (t.start, t.stop) if ctrl else (t.stop, t.postStop)
        en = inRange(en, fi, la)
        tSfx = ("" if ctrl else " post") + ", "
        for j, eni in enumerate(en[:2]):
          i1, i2 = eni-1, eni+1
          imgs.extend(self._annImgs(i1, i2, show='d', ctrl=ctrl))
          for i in range(i1, i2):
            hdr = ""
            if i == i1:
              hdr = "%sf %+d" %(t.sname()+tSfx if j == 0 else "", i-fi)
            elif i == eni:
              hdr = "enter"
            hdrs.append(hdr)
      if imgs:
        img = combineImgs(imgs, hdrs=hdrs, nc=(i2-i1)*2,
          hdrL=basename(self.va.fn))[0]
        fn = CALC_REWARDS_IMG_FILE %("ctrl" if ctrl else "post", self.f+1)
        writeImage(fn, img)

  # calculate rewards (circle enter events)
  # * calculation done for fly 1's actual training circle, fly 2's virtual
  #  training circle, and fly 1 and 2's control circles (if defined)
  # * naming: calculated vs. control reward (see comment at beginning of file)
  def _calcRewards(self):
    if not self.va or not self.va.circle:
      return
    ens = [[], []]   # enter events
    self.no_en = self.no_on = 0   # statistics for mismatch calc. vs. actual
    self.rmImNum, nEnT, nEn0T, twc = 0, [0, 0], 0, []
    for t in self.va.trns:
      x, y = self.xy(t.start, t.postStop)
      for i, (cx, cy, r) in enumerate(t.circles(self.f)):
        dc = np.linalg.norm([x-cx, y-cy], axis=0)
        inC = (dc < r).astype(np.int) + (dc < r+BORDER_WIDTH)
        for s in trueRegions(self.nan[t.start:t.postStop]):
          inC[s] = inC[s.start-1] if s.start > 0 else False
        idxs = np.arange(len(inC))[inC != 1]
        en = idxs[np.flatnonzero(np.diff(inC[inC != 1]) == 2)+1] + t.start
        ctrl = i > 0
        ens[ctrl].append(en)

        if i == 0:
          en0 = (np.diff((inC > 1).astype(np.int)) == 1).nonzero()[0]+1+t.start
          self._checkRewards(t, en0)
          nEn0T += inRange(en0, t.start, t.stop, count=True)
          if BORDER_WIDTH == 0:
            assert np.array_equal(en, en0)
        elif i == 1:
          twc.append(t.n)
        nEnT[ctrl] += inRange(en, t.start, t.stop, count=True)

    self.en = [np.sort(np.concatenate(en)) for en in ens]
      # idx: 0: calculated, 1: control
    nt = nEnT[0]
    print "  total calculated rewards during training: %d" %nt
    if self.f == 0:
      bw0 = BORDER_WIDTH == 0
      if not bw0:
        print "    for zero-width border: %d%s" %(nEn0T,
          "" if nt == 0 else " (+{:.1%})".format((nEn0T-nt)/nt))
      msg = []
      for no, tp in ((self.no_en, "calc."), (self.no_on, "actual")):
        if no:
          msg.append("only %s: %d" %(tp, no))
      print "%s    compared with actual ones: %s" %("" if bw0 else "  ",
        ", ".join(msg) if msg else "identical")
      if msg and opts.showRewardMismatch:
        cv2.waitKey(0)
    print "  total control rewards during training%s %s: %d" %(
      pluralS(len(twc)), commaAndJoin(twc), nEnT[1])

  def _plotIssues(self):
    if not self.va:
      return
    susT, susC, losT, losC = 'suspicious jump', 'w', 'lost', 'y'
    if self.f == 0:
      plt.figure(basename(self.va.fn) + " Tracking Issues")
      plt.imshow(cv2.cvtColor(self.va.frame, cv2.COLOR_BGR2RGB))
      plt.axis('image')
      tx = plt.gca().transAxes
      for x, c, t in ((.25, susC, susT), (.75, losC, losT)):
        plt.text(x, .05, t, color=c, transform=tx, ha='center')
    for ji in self.susp:
      plt.plot(self.x[ji:ji+2], self.y[ji:ji+2], color=susC)
    print "    suspicious jumps: %s" %", ".join(
      "%s (%d)" %(s2time(ji/self.va.fps), ji) for ji in self.susp[::2])
    for r in self.nanrs:
      f, t = r.start, r.stop
      plt.plot(self.x[f:t], self.y[f:t], color=losC, marker='o', ms=3, mew=0)

  # returns list with annotated images for frames in the range [i1, i2)
  # show: see annotateTxt()
  def _annImgs(self, i1, i2, show='', ctrl=False):
    imgs = []
    for i in range(i1, i2):
      img = readFrame(self.va.cap, i)
      t, cpr = Training.get(self.va.trns, i, includePost=True), None
      if t:
        cpr = t.annotate(img, ctrl=ctrl, f=self.f)
      ellDrwn = self.annotate(img, i)
      img = self.va.extractChamber(img)
      self.annotateTxt(img, i, show, cpr)
      # uncomment to show ellipse params:
      # TODO: move to player?
      # if ellDrwn:
      #   (x, y), (w, h), theta = self.ellipse(i)
      #   putText(img, "w = %.1f, h = %.1f, theta = %.1f" %(w, h, theta),
      #     (5,5), (0,1), textStyle(color=COL_W))
      imgs.append(img)
    return imgs

  # draws ellipse for frame i and trajectory of length tlen on the given image,
  #  returning whether ellipse was drawn
  def annotate(self, img, i, tlen=1, col=COL_Y):
    nn = not self.nan[i]
    if nn:
      cv2.ellipse(img, self.ellipse(i), col, 1)
    i1 = max(i-tlen, 0)
    xy = self.xy(i1, i+1)
    xy = [a[~np.isnan(a)] for a in xy]
    if len(xy) > 1:
      cv2.polylines(img, xy2Pts(*xy), False, COL_Y_D)
    return nn

  # annotate with
  # * difference in timestamp between frame i and previous frame ('t' in show)
  # * 'd [<|>=] r' ('d' in show)
  # * experimental fly ('f' in show)
  def annotateTxt(self, img, i=None, show='', cpr=None):
    txt, alrt = [], False
    if i > 0 and 't' in show:
      dt, dt0 = self.ts[i] - self.ts[i-1], 1/self.va.fps
      alrt = abs(dt-dt0)/dt0 > .1
      txt.append('+%.2fs' %dt)
    if cpr and 'd' in show:
      txt.append('d %s r' %(
        '<' if distance(self.xy(i), cpr[:2]) < cpr[2] else '>='))
    if 'f' in show:
      txt.append('%d' %self.va.ef)
    if txt:
      putText(img, ", ".join(txt), (5,5), (0,1),
        textStyle(color=COL_Y if alrt else COL_W))

  # - - -

  @staticmethod
  def _test():
    nan = np.nan
    xy = (np.array(e) for e in (
      [nan, 1, nan, 2, nan], [nan, 2, nan, 4, nan]))
    t = Trajectory(xy)
    requireClose((t.x, t.y), ([nan, 1, 1.5, 2, 2], [nan, 2, 3, 4, 4]))
    requireClose(t.d, [0, np.sqrt(.5**2+1), np.sqrt(.5**2+1), 0, 0])
    requireClose(t.d[1], t.dist(1,2))

  # - - -

  # returns distance traveled between the given frames
  def distTrav(self, i1, i2):
    return np.sum(self.d[i1:i2])

  # returns distance between the given frames or, if no frames given,
  #  distances array, giving distance between current and next frame
  #  (and 0 for frames when tracking had not started yet)
  def dist(self, i1=None, i2=None):
    return self.d if i1 is None else \
      distance((self.x[i1], self.y[i1]), (self.x[i2], self.y[i2]))

  # returns x and y arrays
  def xy(self, i1=None, i2=None):
    return (self.x, self.y) if i1 is None else (
      (self.x[i1], self.y[i1]) if i2 is None else
      (self.x[i1:i2], self.y[i1:i2]))

  # returns RDP-simplified x and y arrays
  def xyRdp(self, i1, i2, epsilon):
    return xy2T(rdp(self.xy(i1, i2), epsilon, _RDP_PKG))

  # returns ellipse for the given frame
  def ellipse(self, i):
    return ((self.x[i], self.y[i]), (self.w[i], self.h[i]), self.theta[i])

  # returns or sets whether trajectory is "bad" (e.g., has too many suspicious
  #  jumps)
  def bad(self, bad=None):
    if bad is not None:
      self._bad = bad
    return self._bad

# - - -

# analysis of a single video
class VideoAnalysis:

  _ON_KEY = re.compile(r'^v[1-9]\d*(\.\d+)?$')   # excludes, e.g., 'v0'

  numPostBuckets, numNonPostBuckets = None, 4
  rpiNumPostBuckets, rpiNumNonPostBuckets = None, 0

  fileCache = pylru.lrucache(1)
  currFn, currAImg = None, None

  # f: fly to analyze, e.g., for HtL (0-19); None: regular chamber
  def __init__(self, fn, gidx, f=None):
    print "=== analyzing %s%s ===\n" %(
      basename(fn), "" if f is None else ", fly %d" %f)

    self.gidx, self.f = gidx, f
    self._loadData(fn)
    self.flies = (0,) if self.noyc else (0, 1)
    self._skipped = True   # for early returns
    if opts.annotate:
      self._writeAnnotatedVideo()
      return
    setOptionDefaults(self)
    self._initTrx()
    self._readNoteFile(fn)   # possibly overrides whether trajectories bad
    if opts.matFile:
      self._writeMatFile()
      return
    elif opts.play:
      self._playAnnotatedVideo()
      return
    if self.trx[0].bad():
      print "\n*** skipping analysis ***"
      return
    print

    self._skipped = False
    if self.circle or self.choice:
      self._analysisImage()
    self.byBucket()
    if self.circle:
      self.bySyncBucket()
      self.bySyncBucket2()   # pass True to get maximum distance reached
      self.byPostBucket()
      self.byReward()
      if opts.plotTrx:
        self.plotTrx()
      if opts.plotThm or opts.thm:
        self.plotTrx(True)
      if opts.rdp:
        self.rdpAnalysis()
      self.speed()
      self.rewardsPerMinute()
    if self.choice:
      if self.openLoop:
        self.posPrefOL()
      else:
        self.posPref()
      if opts.plotTrx:
        if opts.ol:
          self.plotTrx()
        else:
          self.plotYOverTime()
      if opts.ol:
        self.bySyncBucket2(True)
    if opts.move:
      self.distance()
    if opts.hm:
      self.calcHm()

    if opts.delayCheckMult is not None:
      self.delayCheck()

  # returns whether analysis was skipped
  def skipped(self): return self._skipped

  # writes images with some calculated rewards
  def calcRewardsImgs(self):
    for trx in self.trx:
      trx.calcRewardsImg()

  # note: called for non-skipped analysis only
  def _analysisImage(self):
    if self.fn != self.currFn:
      VideoAnalysis.currFn = self.fn
      img = self.aimg = VideoAnalysis.currAImg = self.frame.copy()
    else:
      img, self.aimg = self.currAImg, None
    for t in self.trns:
      t.annotate(img, ctrl=None)

  # extractChamber() extracts the experimental fly's chamber floor plus the
  #  given border from the given frame
  def _createExtractChamber(self):
    (xm, ym), (xM, yM) = self.ct.floor(self.xf, f=self.ef)
    bw = {CT.regular: 0, CT.htl: 15, CT.large: 35}[self.ct]
    def exCh(frame, borderw=bw):
      return subimage(frame, (xm-borderw, ym-borderw), (xM+borderw, yM+borderw))
    self.extractChamber = exCh

  def _loadData(self, fn):
    self.cap = videoCapture(fn)
    self.fps, self.fn = frameRate(self.cap), fn
    self.frame, self.bg = readFrame(self.cap, 0), None

    if fn not in self.fileCache:
      self.fileCache[fn] = [unpickle(replaceCheck(AVI_X, x, fn)) for x in
        (".data", ".trx")]
    self.dt, self.trxRw = self.fileCache[fn]
    x, proto = self.trxRw['x'], self.dt['protocol']
    nfls, self.nf = len(x), len(x[0])
    self.ct = CT.get(nfls)
    self.fns, self.info = (proto[k] for k in ('frameNums', 'info'))
    multEx = isinstance(self.fns, list)
    nef = self.nef = len(self.fns) if multEx else 1
    self.noyc, self.ef = nfls == nef, self.f or 0
    assert self.noyc or nef == int(nfls/2)
    if self.ef >= nef:
      error('fly number %d out of range (only %s)'
        %(self.ef, nItems(nef, "experimental fly")))
    yTop, yBottom = (proto['lines'][k] for k in ('yTop', 'yBottom')) \
      if 'lines' in proto else (None, None)
    if self.f is None:
      if multEx:
        error('more than one experimental fly and no fly numbers; use ' +
          '-v with : or -f')
      assert self.ct == CT.regular
    elif multEx:
      self.fns, self.info = self.fns[self.ef], self.info[self.ef]
      if yTop:
        yTop, yBottom = yTop[self.ef], yBottom[self.ef]
    area, self.pt = 'area' in proto, proto.get('pt')
    self.xf = Xformer(proto.get('tm'), self.ct, self.frame,
      proto.get('fy', False))
    self.circle = area or self.pt == 'circle'
    self.openLoop = self.pt == 'openLoop'
    self.trns, tms = [], zip(self.fns['startTrain'], self.fns['startPost'])
    self.startPre = self.fns['startPre'][0]
      # note: some older experiments used 'startPre' more than once
    if self.circle:
      r = proto['area' if area else 'circle']['r']
      rl = self.info.get('r', [])
      if len(rl) == len(tms):
        r = rl
      else:
        assert all(r1 == r for r1 in rl)
      cPos = self.info['cPos']
    if self.openLoop:
      self.alt = proto.get('alt', True)
    for i, (st, spst) in enumerate(tms):
      if self.circle:
        trn = Training(i+1, st, spst, self,
          circle=(cPos[i], r if np.isscalar(r) else r[i]))
      else:
        trn = Training(i+1, st, spst, self,
          ytb=None if yTop is None else (yTop, yBottom))
      self.trns.append(trn)
    # frame indexes of rewards
    on = [self.fns[k] for k in self.fns if self._ON_KEY.match(k)]
    self.on = np.sort(np.concatenate(on)) if on else np.array([])
    if self.openLoop:
      self.off = np.array(self.fns['v0'])
      assert np.array_equal(self.off, np.sort(self.off))

    print "  video length: %s, frame rate: %s fps, chamber type: %s" %(
      frame2hm(self.nf, self.fps), formatFloat(self.fps, 1), self.ct)
    print "  (pre: %s)" %frame2hm(self.trns[0].start-self.startPre, self.fps)
    Training.processReport(self.trns, self.on, self.nf)
    self.choice = all(t.tp is t.TP.choice for t in self.trns)
      # note: also used for protocol type openLoop
    self._createExtractChamber()

  def _initTrx(self):
    print "\nprocessing trajectories..."
    self.trx, ts = [], self.trxRw.get('ts')
    self.trxf = (self.ef,) if self.noyc else (self.ef, self.ef+self.nef)
    for f in self.trxf:
      x, y, w, h, theta = (np.array(self.trxRw[xy][f]) for xy in
        ('x', 'y', 'w', 'h', 'theta'))
      self.trx.append(Trajectory((x, y), (w, h, theta), len(self.trx),
        va=self, ts=ts))

  # note file
  # * overrides, e.g., suspicious jump exclusion
  # * e.g., "e0,i2": exclude fly 0, include fly 2
  # * fly numbering is yoked control-independent (e.g., fly 0 is experimental
  #  fly for regular chamber)
  _EI_NUM = re.compile(r'^(e|i)(\d+)$')
  def _readNoteFile(self, fn):
    nfn = replaceCheck(AVI_X, "__note.txt", fn)
    note = readFile(nfn)
    if note is not None:
      print "\nreading %s:" %basename(nfn)
      note, ov = note.strip(), False
      for ps in note.split(','):
        mo = self._EI_NUM.match(ps)
        try:
          excl, f1 = mo.group(1) == 'e', int(mo.group(2))
        except:
          error('cannot parse "%s"' %note)
        if f1 in self.trxf:
          f = self.trxf.index(f1)
          if self.trx[f].bad() != excl:
            self.trx[f].bad(excl)
            print "  %scluding fly %d" %("ex" if excl else "in", f+1)
            ov = True
      if not ov:
        print "  no override"

  # - - -

  def _writeAnnotatedVideo(self):
    ofn = replaceCheck(AVI_X, '__ann.avi', self.fn)
    print "\nwriting annotated video %s..." %basename(ofn)
    out = cv2.VideoWriter(ofn, cvFourcc('MJPG'), self.fps,
      imgSize(self.frame), isColor=True)
    i = 0
    setPosFrame(self.cap, i)
    while True:
      ret, frm = self.cap.read()
      if not ret:
        break
      t = Training.get(self.trns, i)
      if t:
        t.annotate(frm)
      out.write(frm)
      i += 1
    out.release()

  def _writeMatFile(self):
    matDir = 'mat'
    if not os.path.exists(matDir):
      os.makedirs(matDir)
    ofn = os.path.join(matDir, basename(replaceCheck(AVI_X, '.mat', self.fn)))
    print "\nwriting MATLAB file %s..." %ofn
    t = []
    for f in (0, 1):
      trx = self.trx[f]
      t.append([[], []] if trx.bad() else self.xf.f2t(trx.x, trx.y))
    d = dict(f1x=t[0][0], f1y=t[0][1], f2x=t[1][0], f2y=t[1][1],
      trainings=np.array([[t.start, t.stop] for t in self.trns])+1,
      on=self.on+1)
    sio.savemat(ofn, d)

  # - - -

  _DLT = 100
  _ARROW_KEY_MAP = {83:1, 84:_DLT, 81:-1, 82:-_DLT,
    ord('.'):1, ord('>'):_DLT, ord(','):-1, ord('<'):-_DLT}
      # note: arrow keys not seen by OpenCV on Mac
  _HLP = re.sub(SPACES_AFTER_TAB, "", textwrap.dedent("""\
    keyboard commands:
    h or ?\t                    toggle show help
    q\t                         quit
    <frame|time> + g\t          go to frame or time (hh:mm:ss)
    <frames|time> + l\t         set length of trajectory shown
    s\t                         toggle show stats
    right, left arrows or .,\t  next, previous frame
    down, up arrows or ><\t     frame +100, -100"""))

  # play video
  def _playAnnotatedVideo(self):
    reg = self.ct is CT.regular
    i = ip = 0
    trx, tlen, s, show, hlp = self.trx, self._DLT, '', False, False
    while True:
      try:
        frm = readFrame(self.cap, i)
      except util.VideoError:
        i = ip
        continue
      ip = i
      t, cpr = Training.get(self.trns, i), None
      if t:
        cpr = t.annotate(frm)
      for trx in self.trx:
        trx.annotate(frm, i, tlen, COL_Y if trx.onBottom[i] else COL_R)
      if reg:
        frm = cv2.resize(frm, (0,0), fx=2, fy=2)
      if show:
        txt = []
        for f, trx in enumerate(self.trx):
          txt1 = []
          txt1.append('f%d:' %(f+1))
          txt1.append('d=%.1f' %trx.d[i])
          txt1.append('ar=%.1f' %trx.ar[i])
          txt1.append('onB=%s' %("T" if trx.onBottom[i] else "F"))
          if reg:
            #txt1.append('dx=%.1f' %trx.dltX[i])
            txt1.append('dx2=%.1f' %trx.dltX2[i])
          txt.append(" ".join(txt1))
        putText(frm, "  ".join(txt), (5,5), (0,1), textStyle(color=COL_W))
      elif hlp:
        putText(frm, self._HLP, (5,5), (0,1), textStyle(color=COL_W))
      else:
        self.trx[0].annotateTxt(frm, i, 'td', cpr)
      hdr = '%s (%d) tlen=%d' %(s2time(i/self.fps), i, tlen)
      img = combineImgs(((frm, hdr),))[0]
      cv2.imshow(basename(self.fn), img)

      # if key "press" (possibly due to auto repeat) happened before waitKey(),
      #  waitKey() does *not* process events and the window is not updated;
      #  the following code makes sure event processing is done
      eventProcessingDone = False
      while True:
        k = cv2.waitKey(1)
        if k == -1: eventProcessingDone = True
        elif eventProcessingDone: break
      k &= 255
      dlt, kc = self._ARROW_KEY_MAP.get(k), chr(k)
      if kc == 'q':
        break
      elif kc in ('h', '?'):
        hlp = not hlp
      elif kc in ('g', 'l'):
        n = None
        if DIGITS_ONLY.match(s):
          n = int(s)
        else:
          try:
            n = int(time2s(s)*self.fps)
          except util.ArgumentError: pass
        if n is not None:
          if kc == 'g': i = n
          else: tlen = n
        s = ''
      elif kc == 's':
        show = not show
      elif kc in '0123456789:':
        s += kc
      elif dlt:
        i += dlt

  # - - -

  _TB = "trajectory bad"

  def _bad(self, f): return self.trx[0 if f is None else f].bad()

  # returns frame indexes of all rewards during the given training
  # note: post not used
  def _getOn(self, trn, calc=False, ctrl=False, f=None, post=False):
    on = self.trx[f].en[ctrl] if calc else self.on
    if trn is None:
      return on
    fi, la = (trn.stop, trn.postStop) if post else (trn.start, trn.stop)
    return inRange(on, fi, la)

  # returns number of rewards in the given frame index range
  def _countOn(self, fi, la, calc=False, ctrl=False, f=None):
    on = self._getOn(None, calc, ctrl, f)
    return inRange(on, fi, la, count=True)

  # returns number of rewards by bucket; fiCount can be used to make
  #  counting start later than fi
  def _countOnByBucket(self, fi, la, df, calc=False, ctrl=False, f=None,
      fiCount=None):
    nOns, fi0 = [], fi
    while fi+df <= la:
      nOns.append(self._countOn(
        fi if fiCount is None else max(fi, fiCount), fi+df, calc, ctrl, f))
      fi += df
    if fiCount is None:
      assert sum(nOns) == self._countOn(fi0, fi, calc, ctrl, f)
    return nOns

  # returns frame index of first reward in the given frame index range
  def _idxFirstOn(self, fi, la, calc, ctrl, f=0):
    on = inRange(self._getOn(None, calc, ctrl, f), fi, la)
    return on[0] if len(on) else None

  # returns frame index of first frame where fly 0 is on control side (across
  #  midline) in the given frame range
  def _idxFirstCtrlSide(self, fi, la, trn):
    yc, ym, ys = trn.circles()[0][1], trn.cntr[1], self.trx[0].y[fi:la]
    assert abs(yc-ym) > trn.r
    onCs = ys > ym if yc < ym else ys < ym
    idx = np.argmax(onCs)
    return fi+idx if onCs[idx] else None

  # returns whether the first reward in first bucket for fly 0 is control
  def _firstRewardCtrl(self, fi, la, df):
    if fi is None or fi+df > la:   # consistent with _countOnByBucket()
      return None
    calc = True
    ic, inc = (self._idxFirstOn(fi, fi+df, calc, ctrl)
      for ctrl in (True, False))
    return (None if inc is None else 0) if ic is None else (
      1 if inc is None else int(ic < inc))

  # returns whether fly 0 crossed midline before first reward in first bucket
  def _xedMidlineBefore(self, fi, la, df, trn):
    if fi is None or fi+df > la or not trn.hasSymCtrl():
        # consistent with _countOnByBucket()
      return None
    on1 = self._idxFirstOn(fi, fi+df, calc=True, ctrl=False)
    im = self._idxFirstCtrlSide(fi, fi+df, trn)
    return (None if on1 is None else 0) if im is None else (
      1 if on1 is None else int(im < on1))

  # appends n of the given values to "to"
  def _append(self, to, vals, f=0, n=2):
    if np.isscalar(vals) or vals is None:
      n, vals = 1, [vals]
    else:
      n = int(n)
    t = n*(np.nan,) if self._bad(f) else \
      tuple(vals[:n]) + (n-len(vals))*(np.nan,)
    assert len(t) == n
    to.append(t)

  def _min2f(self, m): return intR(m*60*self.fps)
  def _f2min(self, a): return a/(60*self.fps)
  def _f2ms(self, a): return time2str(a/self.fps, '%M:%S', utc=True)

  def _printBucketVals(self, vs, f, msg=None, nParen=0, prec=None):
    if prec is not None:
      frm = "%%.%df" %prec
      vs = [frm %v for v in vs]
    vs = ["(%s)" %v if i < nParen else v for i, v in enumerate(vs)]
    print "  %s%s" %("%s: " %msg if msg else "",
      self._TB if self._bad(f) else
        (join(", ", vs, 10) if vs else "no full bucket"))

  def _rewardType(self, calc, ctrl, f):
    return "%s%s" %(cVsA(calc, ctrl), " f%d" %(f+1) if calc or ctrl else "")

  # returns bucket length in frames as int
  def _numRewardsMsg(self, sync):
    blm = opts.syncBucketLenMin if sync else opts.postBucketLenMin
    print "\nnumber%s rewards by %s bucket (%s min):" %(
      "" if sync else " "+cVsA_l(True), "sync" if sync else "post",
      formatFloat(blm, 1))
    return self._min2f(blm)

  # default: skip frame of first reward
  def _syncBucket(self, trn, df=np.nan, skip=1):
    on = self._getOn(trn)   # sync buckets determined using actual rewards
    fi = on[0]+skip if len(on) else None
    if SYNC_CTRL:
      fi = fi if fi is None else noneadd(
        self._idxFirstOn(fi, trn.stop, calc=True, ctrl=True), skip)
    n = np.ceil(trn.len()/df - 0.01)
    return fi, n, on

  # returns SyncType (tp)-dependent frame index in the given frame index range
  # note: skip applies only to sync on control circle
  def _idxSync(self, tp, trn, fi, la, skip=1):
    if tp is ST.fixed or fi is None or np.isnan(fi):
      return fi
    elif tp is ST.control or not trn.hasSymCtrl():
      return noneadd(self._idxFirstOn(fi, la, calc=True, ctrl=True), skip)
    else:
      assert tp is ST.midline
      return self._idxFirstCtrlSide(fi, la, trn)

  # returns start frame of first post bucket
  def _postSyncBucket(self, trn, skip=1):
    return self._idxSync(POST_SYNC, trn, trn.stop, trn.postStop, skip)

  # - - -

  # number of rewards by bucket
  def byBucket(self):
    tnOn = 0
    for i, t in enumerate(self.trns):
      df = t.len()/opts.numBuckets
      if opts.showByBucket:
        if i == 0:
          print "number rewards: (bucket: %s)" %frame2hm(df, self.fps)
        print t.name()
      la, nOns = t.start, []
      for i in range(opts.numBuckets):
        fi, la = la, t.start + intR((i+1)*df)
        nOns.append(self._countOn(fi, la))
      snOn = sum(nOns)
      assert la == t.stop and self._countOn(t.start, t.stop) == snOn
      tnOn += snOn
      if opts.showByBucket:
        print "  %s  (sum: %d)" %(", ".join(map(str, nOns)), snOn)

    print "total rewards training: %d, non-training: %d" %(
      tnOn, len(self.on) - tnOn)
    self.totalTrainingNOn = tnOn

  # number of rewards by sync bucket
  def bySyncBucket(self):
    df = self._numRewardsMsg(True)
    self.numRewards = [[[]], [[], []]]   # idxs: calc, ctrl
    self.rewardPI, self.rewardPITrns = [], []
    self.firstRewardCtrl, self.xedMidlineBefore = [], []
    for t in self.trns:
      print t.name()
      fi, n, on = self._syncBucket(t, df)
      la = min(t.stop, int(t.start+n*df))
      fiRi = none2val(self._idxSync(RI_START, t, fi, la), la)
      self.rewardPITrns.append(t)
      self._append(self.firstRewardCtrl, self._firstRewardCtrl(fi, la, df))
      self._append(self.xedMidlineBefore, self._xedMidlineBefore(fi, la, df, t))
      for calc, f in ((False, None), (True, 0), (True, 1)):
        if self.noyc and f == 1: continue
        for ctrl in ((False, True) if calc else (False,)):
          nOns = [] if fi is None else self._countOnByBucket(
            fi, la, df, calc, ctrl, f, fiRi if calc else None)
          self._printBucketVals(nOns, f, msg=self._rewardType(calc, ctrl, f))
          self._append(self.numRewards[calc][ctrl], nOns, f)
          if ctrl:
            pis = prefIdx(nOnsP, nOns, n=opts.piTh)
            self._printBucketVals(pis, f, msg="  PI", prec=2)
            self._append(self.rewardPI, pis, f, n=n)
          nOnsP = nOns

  # distance traveled or maximum distance reached between (actual) rewards
  #  by sync bucket
  # notes:
  # * reward that starts sync bucket included here (skip=0) so that
  #  distance to the next reward is included in average; this differs from
  #  bySyncBucket() but matches byActualReward()
  # * also used for "open loop" analysis, where sync buckets equal buckets
  def bySyncBucket2(self, maxD=False):
    hdr = "\naverage %s between actual rewards by %sbucket:" %(
      "maximum distance reached" if maxD else "distance traveled",
      "" if opts.ol else "sync ")
    print hdr
    self.bySB2Header, self.bySB2 = hdr, []
    df = self._min2f(opts.syncBucketLenMin)
    for t in self.trns:
      print t.name()
      fi, n, on = self._syncBucket(t, df, skip=0)
      assert not opts.ol or fi == t.start
      la = min(t.stop, t.start+n*df)
      nOns, adb = [], [[], []]
      if fi is not None:
        nOns1 = self._countOnByBucket(fi, la, df)
        while fi+df <= la:
          onb = inRange(on, fi, fi+df)
          nOn = len(onb)
          for f in self.flies:
            if maxD:
              maxDs = []
              for i, f1 in enumerate(onb[:-1]):
                xy = self.trx[f].xy(f1, onb[i+1])
                maxDs.append(np.max(distances(xy, True)))
              adb[f].append(np.nan if nOn < opts.adbTh else np.mean(maxDs))
            else:
              adb[f].append(np.nan if nOn < opts.adbTh else
                self.trx[f].distTrav(onb[0], onb[-1])/(nOn-1))
          nOns.append(len(onb))
          fi += df
        assert nOns == nOns1
      for f in self.flies:
        self._printBucketVals(adb[f], f, msg="f%d" %(f+1), prec=1)
        self._append(self.bySB2, adb[f], f, n=n if opts.ol else n-1)

  def byPostBucket(self):
    self.positionalPiPost()
    self.calcRewardsPost()
    self.rewardPiPost()

  FRAC_OF_BUCK_FOR_PI = 0.05
  def positionalPiPost(self):
    blm, rm = opts.piBucketLenMin, opts.radiusMult
    df = self._min2f(blm)
    self.posPI, self.posPITrns = [], []
    print "\npositional PI (r*%s) by post bucket (%s min):" \
      %(formatFloat(rm, 2), formatFloat(blm, 1))

    trx = self.trx[0]   # fly 1
    (x, y), bad = trx.xy(), trx.bad()
    assert not bad
    for t in self.trns:
      if not t.hasSymCtrl():
        continue
      fi, la, pis, r = t.stop, t.postStop, [], t.r*rm
      print "%s (total post: %s)" %(t.name(), frame2hm(la-fi, self.fps))
      while fi+df <= la:
        xb, yb = x[fi:fi+df], y[fi:fi+df]
        nf = [np.count_nonzero(
          np.linalg.norm([xb-cx, yb-cy], axis=0) < r) for (cx, cy, __) in
            t.circles()]
        nfsum = sum(nf)
        pis.append(np.nan if nfsum < self.FRAC_OF_BUCK_FOR_PI*df else
          (nf[0] - nf[1])/nfsum)
        fi += df
      self._printBucketVals(["%.2f" %pi for pi in pis], f=0)
      self.posPITrns.append(t)
      self.posPI.append((pis[0] if pis and not bad else np.nan,))

  def calcRewardsPost(self):
    calc, ctrl, nnpb = True, False, self.numNonPostBuckets
    df = self._numRewardsMsg(False)
    self.numRewardsPost, self.numRewardsPostPlot = [], []
    for i, t in enumerate(self.trns):
      print t.name() + (
        "  (values in parentheses are still training)"
        if i == 0 and nnpb > 0 else "")
      for f in self.flies:
        nOns = self._countOnByBucket(
          t.stop - df*nnpb, t.postStop, df, calc, ctrl, f)
        if self.numPostBuckets is None:
          VideoAnalysis.numPostBuckets = len(nOns)
        nOns1 = nOns[nnpb-1:]
        self._printBucketVals(nOns1, f, msg=self._rewardType(calc, ctrl, f),
          nParen=1)
        self._append(self.numRewardsPost, nOns1, f, n=4)
        self._append(self.numRewardsPostPlot, nOns, f, n=self.numPostBuckets)

  def rewardPiPost(self):
    calc, blm, nnpb = True, opts.rpiPostBucketLenMin, self.rpiNumNonPostBuckets
    print "\nreward PI by post %sbucket (%s min)" %(
      "" if POST_SYNC is ST.fixed else "sync ", formatFloat(blm, 1))
    df = self._min2f(blm)
    self.rewardPiPst = []
    for i, t in enumerate(self.trns):
      print t.name() + (
        "  (values in parentheses are still training)"
        if i == 0 and nnpb > 0 else "")
      pfi = none2val(self._postSyncBucket(t), t.postStop)
      fiRi = none2val(self._idxSync(RI_START_POST, t, pfi, t.postStop),
        t.postStop)
      for f in self.flies:
        nOns = []
        for ctrl in (False, True):
          nOns.append(concat(
            self._countOnByBucket(fi, la, df, calc, ctrl, f, fiC)
            for fi, la, fiC in ((t.stop - df*nnpb, t.stop, None),
              (pfi, t.postStop, fiRi))))
        pis = prefIdx(nOns[0], nOns[1], n=opts.piTh)
        if self.rpiNumPostBuckets is None:
          VideoAnalysis.rpiNumPostBuckets = nnpb + \
            (t.len(post=True) - 3*self.fps)//df
        self._printBucketVals(pis, f, msg="f%d" %(f+1), prec=2, nParen=nnpb)
        self._append(self.rewardPiPst, pis, f, n=self.rpiNumPostBuckets)

  # - - -

  # analyzes, e.g., time between rewards
  def byReward(self):
    self.byActualReward()
    self.byCalcReward()

  def _byRewardMsg(self, calc):
    nrc = opts.numRewardsCompare
    print "\nby %s reward: (first %d vs. next %d)" %(cVsA_l(calc), nrc, nrc)
    return nrc

  def _plot(self, sp, data, title, xlabel, ylabel, ylim, f=None):
    def xrng(ys, off=0): return range(1+off, len(ys)+1+off)
    ax = plt.subplot(*sp)
    if f != 1:
      plt.plot(xrng(data), data, color='0.5')
    for i, (n, c) in enumerate(
        ((25, (.5, .5, 0)), (50, 'g'), (100, 'b'), (200, 'r'))):
      if f == 1 and n != 100:
        continue
      if len(data) > n:
        avgs = np.convolve(data, np.ones(n)/n, mode='valid')
        plt.plot(xrng(avgs), avgs, color=c, linestyle='--' if f == 1 else '-')
        if sp[2] == 1:
          plt.text(.75, .85-i*.08, 'n = %d' %n, color=c, transform=ax.transAxes)
    if title:
      plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)

  def _firstNVsNext(self, data, n, lbl, appendTo, f=None):
    bad = self._bad(f)
    a = tuple(np.mean(data[i*n:(i+1)*n]) if not bad and (i+1)*n <= len(data)
      else np.nan for i in range(2))
    appendTo.append(a)
    print "  avg. %s%s: %s" %(lbl,
      "" if f is None else " (f%d)" %(f+1),
      "trajectory bad" if bad else "%.1f vs. %.1f" %a)

  def _distTrav(self, f, on):
    trx, db = self.trx[f], []
    for fi, la in zip(on[:-1], on[1:]):
      db.append(trx.distTrav(fi, la))
    assert not db or np.isclose(sum(db), trx.distTrav(on[0], on[-1]))
    return db

  def byActualReward(self):
    nrc = self._byRewardMsg(False)
    self.avgTimeBetween, self.avgDistBetween = [], []
    if opts.showPlots:
      plt.figure(basename(self.fn), (20, 10))
    for i, t in enumerate(self.trns):
      print t.name()
      tnl, xlbl = t.name(short=False), 'reward'
      on = self._getOn(t)
      nr = len(on) if opts.plotAll else nrc*2+1
      on1 = on[:nr]

      ylbl = 'time between [s]'
      tb = np.diff(on1)/self.fps
      self._firstNVsNext(tb, nrc, ylbl, self.avgTimeBetween)
      if opts.showPlots:
        self._plot((2, 3, 1+i), tb, tnl, xlbl, ylbl, (0, 40))

      ylbl = 'distance between'
      for f in self.flies:
        db = self._distTrav(f, on1)
        self._firstNVsNext(db, nrc, ylbl, self.avgDistBetween, f)
        if opts.showPlots and not bad:
          self._plot((2, 3, 4+i), db, None, xlbl, ylbl, (0, 1600), f)

  def byCalcReward(self):
    nrc = self._byRewardMsg(True)
    self.avgTimeBtwnCalc, self.avgDistBtwnCalc = [], []
    for t in self.trns:
      print t.name()
      for f in self.flies:
        on = self._getOn(t, True, f=f)[:nrc*2+1]
        tb = np.diff(on)/self.fps
        self._firstNVsNext(tb, nrc, 'time between [s]',
          self.avgTimeBtwnCalc, f)
      for f in self.flies:
        on = self._getOn(t, True, f=f)[:nrc*2+1]
        db = self._distTrav(f, on)
        self._firstNVsNext(db, nrc, 'distance between',
          self.avgDistBtwnCalc, f)

  # - - -

  # returns normalized trajectory (starting in orig, rotated to go up) for
  #  the given trajectory tuple
  def _normalize(self, xy, orig):
    xy = xy2M(xy)
    sxy = rdp(xy, opts.rdp, _RDP_PKG)
    ra = 0 if len(sxy) < 2 else normAngles(-np.pi/2-velocityAngles(sxy)[0])
    nxy = np.array(sa.rotate(
      sa.translate(sg.LineString(xy), orig[0]-xy[0][0], orig[1]-xy[0][1]),
        ra, origin=orig, use_radians=True))
    return xy2T(nxy)

  # plots trajectories either individually (if hm is False) or by normalizing
  #  them and combining them in heatmap
  def plotTrx(self, hm=False):
    print "\nwriting trajectory images..."
    df, fn = self._min2f(opts.syncBucketLenMin), basename(self.fn, False)
    self.avgMaxDist, self.avgFirstTA, self.avgFirstRL = ([[], []] for
      i in range(3))
    self.ntrx, bmax = 24, 0
    if hm:
      assert opts.rdp   # see TODO at top
      w, h = imgSize(self.frame)
      img1 = getImg(2*h, 2*w, 1, 0)
      def center(f): return intR((.5+f)*w, h)
    for t in self.trns:
      fi, n, on = self._syncBucket(t, df, skip=0)
      f0, b = fi, 1
      while fi+df <= t.stop:
        if t.n == 1 and bmax < b:   # hack: learn bmax in first training
          bmax = b
        f1, imgs, hdrs = None, [], []
        if hm:
          mp = np.ones((2*h, 2*w), np.float32)
          maxDs, turnAs, runLs = [[], []], [[], []], [[], []]
        for f2 in inRange(on, fi, fi+df)[:(self.ntrx+1)]:   # single batch
          if f1:
            if not hm:
              try:
                img = readFrame(self.cap, f2)
              except util.VideoError:
                print "could not read frame %d" %f2
                img = self.frame.copy()
                pass
              t.annotate(img, col=COL_BK)
            txt = []
            for f in self.flies:
              trx = self.trx[f]
              xy = trx.xy(f1, f2+1)
              if hm:
                maxDs[f].append(np.max(distances(xy, True)))
                sxy = trx.xyRdp(f1, f2+1, epsilon=opts.rdp)
                tas = turnAngles(sxy)
                if len(tas):
                  turnAs[f].append(tas[0])
                rls = distances(sxy)
                if len(rls):
                  runLs[f].append(rls[0])
                xy = self._normalize(xy, center(f))
                img1[...] = 0
                cv2.polylines(img1, xy2Pts(*xy), False, 1)
                mp += img1
              else:
                pts = xy2Pts(*xy)
                cv2.polylines(img, pts, False, COL_W)
                cv2.circle(img, tuple(pts[0,-1,:]), 3, COL_W, -1)
                if opts.rdp:
                  sxy = trx.xyRdp(f1, f2+1, epsilon=opts.rdp)
                  spts = xy2Pts(*sxy)
                  cv2.polylines(img, spts, False, COL_Y)
                  for i in range(1, spts.shape[1]-1):
                    cv2.circle(img, tuple(spts[0,i,:]), 2, COL_Y, -1)
                  tas = turnAngles(sxy)
                  txt.append("ta0 = %s" %(
                    "%.1f" %(tas[0]*180/np.pi) if len(tas) else "NA"))
            if not hm:
              if txt:
                putText(img, ", ".join(txt), (5,5), (0,1),
                  textStyle(color=COL_W))
              imgs.append(img)
              hdrs.append("%s (%d-%d)" %(self._f2ms(f2-f0), f1, f2))
          f1 = f2
        if hm:
          img = heatmap(mp)
          for f in self.flies:
            # start circle
            c = center(f)
            cv2.circle(img, c, 3, COL_W, -1)
            # average max distance
            amd = np.mean(maxDs[f])
            r = intR(amd)
            cv2.circle(img, c, r, COL_W)
            # center of mass (inside average max distance)
            mp1 = mp - 1
            msk = np.zeros_like(mp1, dtype=np.uint8)
            cv2.circle(msk, c, r, 1, -1)
            mp1[msk == 0] = 0
            com = ndi.measurements.center_of_mass(mp1)
            # for debugging:
            # print msk[h-5:h+5,f*w+w/2-5:f*w+w/2+5]
            cv2.circle(img, intR(com[::-1]), 3, COL_O, -1)
            # turn angles and run lengths
            atad = arl = None
            if turnAs[f] and runLs[f]:
              ata, arl = np.mean(np.abs(turnAs[f])), np.mean(runLs[f])
              atad = ata*180/np.pi
              c = tupleAdd(c, (0, h/2))
              cv2.line(img, c,
                intR(c[0]+arl*np.sin(ata), c[1]-arl*np.cos(ata)), COL_W)
            if b <= bmax:
              self.avgMaxDist[f].append(amd)
              self.avgFirstTA[f].append(atad)
              self.avgFirstRL[f].append(arl)
          if opts.plotThm:
            cv2.imwrite(TRX_IMG_FILE2 %(fn, t.n, b, "_hm"), img)
        else:
          img = combineImgs(imgs, hdrs=hdrs, nc=6)[0]
          cv2.imwrite(TRX_IMG_FILE2 %(fn, t.n, b, ""), img)
        b += 1
        fi += df

  # - - -

  # analyze after RDP simplification
  def rdpAnalysis(self):
    blm, eps, t = 10, opts.rdp, self.trns[-1]
    print "\nanalysis after RDP simplification (epsilon %.1f)" %eps
    self.rdpInterval = "last %s min of %s" %(formatFloat(blm, 1), t.name())
    print self.rdpInterval
    assert self.circle and len(self.trns) == 3 and t.tp is t.TP.center
    self.rdpAvgLL, self.rdpTA = [], []
    on = self._getOn(t)
    f1, d, ta = None, [[], []], [[], []]
    for f2 in inRange(on, t.stop-self._min2f(blm), t.stop):
      if f1:
        for f in self.flies:
          sxy = self.trx[f].xyRdp(f1, f2+1, epsilon=eps)
          d[f].extend(distances(sxy))
          ta[f].append(turnAngles(sxy))
      f1 = f2
    print "avg. line length"
    for f in self.flies:
      mll = np.mean(d[f]) if len(d[f]) >= RDP_MIN_LINES else np.nan
      print "  f%d: %.1f" %(f+1, mll)
      self._append(self.rdpAvgLL, mll, f)
    print "turn analysis"
    for f in self.flies:
      nt, ndc = 0, 0
      for ta1 in ta[f]:
        tas = np.sign(ta1)
        assert np.count_nonzero(tas) == len(tas) == len(ta1)
          # note: RDP should guarantee there are no 0-degree turns
        nt += len(tas)
        ndc += np.count_nonzero(np.diff(tas))
      print "  f%d: same direction: %s  number turns: %d" %(f+1,
        "{:.2%}".format((nt-ndc)/nt) if nt else "-", nt)
      self.rdpTA.append(None if self._bad(f) else ta[f])

  # - - -

  # calculate chamber background
  # note: used for both heatmaps and LED detector; only one background saved
  #  currently (correct only if heatmaps and LED detector not used together)
  def background(self, channel=BACKGROUND_CHANNEL, indent=0):
    if self.bg is None:
      print " "*indent + "calculating background (channel: %d)..." %channel
      n, nf, nmax, frames = 0, 11, self.trns[-1].postStop, []
      dn = nmax*.8/nf
      for i in range(nf):
        n += random.randint(intR(.2*dn), intR(1.8*dn))
        frames.append(toChannel(readFrame(self.cap, min(n, nmax-1)), channel))
      self.bg = np.median(frames, axis=0)
    return self.bg

  # note: assumes template coordinates
  #  e.g., for large chamber w/out yoked controls, mirror() makes flies 1-3
  #   look like fly 0
  # TODO: replace with Xformer's built-in _mirror()?
  def mirror(self, xy):
    if self.ct is CT.large:
      return [2*268-xy[0] if self.ef%2 else xy[0],
        2*268-xy[1] if self.noyc and self.ef>1 else xy[1]]
    else:
      return xy

  # calculate maps for heatmaps
  def calcHm(self):
    self.heatmap, self.heatmapPost = [[], []], [[], []]   # index: fly, training
    self.heatmapOOB = False
    for i, t in enumerate(self.trns):
      for f in self.flies:
        if self.ct is CT.regular:
          xym = np.array(((-30, 108)[f], -24))
          xyM = np.array(((90, 228)[f], 164))
        elif self.ct is CT.large:
          sw = 36
          xym = np.array((4-sw, (4-sw, 286)[f]))
          xyM = np.array((250, (250, 532+sw)[f]))
        else:
          error('heatmap not yet implemented')
        bins, rng = (xyM - xym)/HEATMAP_DIV, np.vstack((xym, xyM)).T
        trx = self.trx[f]
        for j, hm in enumerate((self.heatmap, self.heatmapPost)):
          if j == 0:
            fi, la, skip = t.start, t.stop, False
          else:
            # note: should there be limit how late fi can be?
            fi = none2val(self._postSyncBucket(t, skip=0))
            la = fi + self._min2f(opts.rpiPostBucketLenMin)
            fiRi = none2val(self._idxSync(RI_START_POST, t, fi, la, skip=0), la)
            skip = not la <= t.postStop   # correct also if la is NaN
          if trx.bad() or skip:
            hm[f].append((None, None, xym))
            continue
          xy = self.mirror([a[fi:la] for a in self.xf.f2t(trx.x, trx.y)])
          for a, m, M in zip(xy, xym, xyM):
            if not (m < np.nanmin(a) and np.nanmax(a) < M):
              self.heatmapOOB = True
            if j:
              a[0:fiRi-fi] = np.nan
          xy = [a[trx.walking[fi:la]] for a in xy]
          assert np.array_equal(np.isnan(xy[0]), np.isnan(xy[1]))
          xy = [a[~np.isnan(a)] for a in xy]
            # due to interpolation, there should be no NaNs due to lost flies
          mp = np.histogram2d(xy[0], xy[1], bins=bins, range=rng)[0]
          hm[f].append((mp.T, la-fi, xym))

  # - - -

  # positional preference
  def posPref(self):
    blm, numB = opts.piBucketLenMin, opts.numBuckets
    print "\npositional preference (for top), including " + \
      formatFloat(blm, 1) + "-min post buckets:"
    if opts.skip:
      print "  " + skipMsg()
    self.posPI, sf = [], self._min2f(opts.skip)
    for t in self.trns:
      print t.name()
      for f in self.flies:
        fi, la, df = t.start, t.postStop, t.len()/numB
        pis, o = [], []
        while fi+df <= la:
          fiI, skip = intR(fi), False
          ivs = ([(fiI, fiI+sf)] if opts.skip and opts.skipPI else []) + \
            [(fiI+sf, intR(fi+df))]
          for i, (f1, f2) in enumerate(ivs):
            y = self.trx[f].y[f1:f2]
            inT, inB = y<t.yTop, y>t.yBottom
            vt, vb = (len(trueRegions(a)) for a in (inT, inB))
            nt, nb = (np.count_nonzero(a) for a in (inT, inB))
            if i == len(ivs)-1:
              skip |= vt < opts.minVis or vb < opts.minVis
            if len(ivs) > 1 and i == 0:
              skip |= nt == 0 or nb == 0
          pi = prefIdx(nt, nb)
          pis.append(np.nan if skip else pi)
          o.append("%s%.2f" %("post: " if len(o) == numB else "", pi))
          fi += df
          if len(o) == numB:
            df = self._min2f(blm)
            assert np.isclose(fi, t.stop)
        self._append(self.posPI, pis, f, n=2)
        print "  f%d: %s" %(f+1, ", ".join(o))

  # positional preference for open loop protocols (both on-off and alternating
  #  side)
  def posPrefOL(self):
    print "\npositional preference for LED side:"
    self.posPI = []
    for t in self.trns:
      print t.name()
      assert t.yTop == t.yBottom
      ivs = ((self.startPre+1, t.start), (t.start, t.stop))
        # y coordinate of trajectory can be NaN for frame startPre
      on = self._getOn(t)
      if not self.alt:
        off = inRange(self.off, t.start, t.stop)
      img = self.extractChamber(readFrame(self.cap, on[0]+2))
      if self.ct is not CT.regular:
        self.trx[0].annotateTxt(img, show='f')
      self.olimg = img
      assert on[0]+1 < on[1] and on[0] <= t.start+1 and on[-1] <= t.stop
      for f in self.flies:
        with np.errstate(invalid='ignore'):   # suppress warnings due to NaNs
          inT, pis = self.trx[f].y<t.yTop, []
        if self.alt:
          for i in range(1, len(on), 2):
            inT[on[i]:on[i+1] if i+1<len(on) else t.stop] ^= True
        else:
          mask = np.zeros_like(inT, dtype=int)
          mask[on] = 1
          mask[off] = -1
          mask = np.cumsum(mask)
          assert mask.min() == 0 and mask.max() == 1
        for i, (f1, f2) in enumerate(ivs):
          inT1, pre, onOff = inT[f1:f2], i == 0, i == 1 and not self.alt
          useMask = pre or onOff
            # for HtL, tracking may not have started at beginning of pre period
          if useMask:
            mask1 = ~np.isnan(self.trx[f].y[f1:f2]) if pre else mask[f1:f2]
          for j in range(2 if onOff else 1):
            if self.trx[f].bad():
              pis.append(np.nan)
            else:
              assert pre or j == 1 or not np.any(np.isnan(self.trx[f].y[f1:f2]))
              if j == 1:
                mask1 ^= 1
              nt = np.count_nonzero(inT1 & mask1 if useMask else inT1)
              nb = (np.count_nonzero(mask1) if useMask else f2-f1) - nt
              pis.append(prefIdx(nt, nb))
        self._append(self.posPI, pis, f, n=2 if self.alt else 3)
        print "  f%d: %.2f (pre), %.2f%s" %(f+1, pis[0], pis[1],
          "" if self.alt else " (on), %.2f (off)" %pis[2]) 

  def plotYOverTime(self):
    df, nr, fn = self._min2f(opts.piBucketLenMin), 4, basename(self.fn, False)
    ledC = '#70e070' if opts.green else '#ff8080'
    for t in self.trns:
      assert t.ct is CT.regular
      plt.figure(figsize=(20, 4*nr))
      yc = self.xf.t2fY(70)
      for f in self.flies:
        fi, la = t.start, t.stop
        dm = max(abs(y-yc) for y in minMax(self.trx[f].y[t.start:t.postStop]))
        ymm = (yc-dm, yc+dm)
        for post in (False, True):
          plt.subplot(nr, 1, 1+2*f+post)
          plt.yticks([])
          plt.ylim(ymm[::-1])
          if post:
            fi, la = t.stop, min(t.stop+df, t.postStop)
          x = self._f2min(np.arange(fi, la))
          xmm = x[[0,-1]]
          plt.xlim(xmm)
          y = self.trx[f].y[fi:la]
          for e in self._f2min(inRange(self.on, fi, la)):
            plt.plot((e,e), ymm, color=ledC)
          plt.plot(x, y, color='.2')
          if hasattr(t, 'yTop'):
            for y in (t.yTop, t.yBottom):
              plt.plot(xmm, (y,y), color='.5', ls='--')
          plt.title("post" if post else
            "fly %d%s" %(f+1, " [%s]" %t.name() if f == 0 else ""))
      plt.savefig(TRX_IMG_FILE %(fn, t.n), bbox_inches='tight')
      plt.close()

  # - - -

  def distance(self):
    numB = opts.numBuckets
    print "\ndistance traveled:"
    for t in self.trns:
      print t.name()
      df = t.len()/numB
      for f in self.flies:
        la, ds, trx = t.start, [], self.trx[f]
        for i in range(numB):
          fi, la = la, t.start + intR((i+1)*df)
          ds.append(trx.distTrav(fi, la))
        td = sum(ds)
        assert np.isclose(trx.distTrav(t.start, t.stop), td)
        self._printBucketVals(ds, f, "f%d (%.0f)" %(f+1, td), prec=0)

  # - - -

  # speed stats
  def speed(self):
    preLenMin, spMinNFrms, bt = 10, 100, SPEED_ON_BOTTOM
    print "\nspeed stats (with values for " + \
      formatFloat(preLenMin, 1) + "-min pre period first):"
    df = self._min2f(preLenMin)
    self.speed, self.stopFrac = [], []
    self.speedLbl = "speed %s[%s/s]" %(
      "bottom " if bt else "", "mm" if bt else "px")
    fi = 0
    for t in self.trns:
      print t.name()
      # check whether pulse in pre period
      on = inRange(self.on, fi, t.start)
      pls = on[-1] if len(on) else t.start
      assert len(on) <= 1   # at most one pulse in pre period
      fi = t.stop + 1   # pulse can happen on t.stop frame
      for f in self.flies:
        trx = self.trx[f]
        sps, stpFs = [], []
        for pre in (True, False):
          f1, f2 = (pls-df, pls) if pre else (t.start, t.stop)
          sp1 = trx.sp[f1:f2]
          if bt:
            sp1 = sp1[trx.onBottomPre[f1:f2]] / trx.pxPerMmFloor
            #print ">>>", t.n, f, pre, len(sp1)
          sps.append(np.nan if len(sp1) < spMinNFrms else np.mean(sp1))
          nw, df12 = np.count_nonzero(trx.walking[f1:f2]), f2 - f1
          stpFs.append((df12-nw)/df12)
        print "  f%d: avg. %s: %s, stop fraction: %s" %(f+1,
          self.speedLbl, join(", ", sps, p=1), join(", ", stpFs, p=2))
        self._append(self.speed, sps, f)
        self._append(self.stopFrac, stpFs, f)

  # rewards per minute
  def rewardsPerMinute(self):
    self.rewardsPerMin = []
    for t in self.trns:
      fi, la = self._syncBucket(t, skip=0)[0], t.stop
      rpm = np.nan if fi is None else self._countOn(
        fi, la, calc=True, ctrl=False, f=0)/self._f2min(la-fi)
      self._append(self.rewardsPerMin, rpm, f=0)

  # - - -

  def initLedDetector(self):
    v, ch = 2, 2   # version (1 or 2)
    assert v in (1, 2)
    (xm, ym), (xM, yM) = self.ct.floor(self.xf, f=self.ef)
    if v == 1:
      k, bg = 1, 0
      print '  algorithm: max (channel: %d)' %ch
    else:
      k = 10
      print '  algorithm: background difference, kth-highest value (k=%d)' %k
      bg = self.background(channel=ch, indent=2)[ym:yM,xm:xM]
    k1 = (xM-xm)*(yM-ym) - k
    # closure stores, e.g., which part of frame to use
    def feature(frame):
      return np.partition(frame[ym:yM,xm:xM,ch] - bg, k1, axis=None)[k1]
    self.feature = feature
    print '  reading frames to learn "LED off"...'
    vs = [self.feature(readFrame(self.cap, n+20)) for n in range(100)]
    self.ledOff = np.mean(vs)
    self.ledTh = self.ledOff + opts.delayCheckMult*np.std(vs)

  # returns combined image if no key given; otherwise, memorizes the given
  #  frame sequence and increments c[key]
  def _delayImg(self, i1=None, i2=None, key=None, c=None):
    if not hasattr(self, '_dImgs'):
      self._dImgs, self._dHdrs = {}, {}
      self._dNc = None if i1 is None else i2-i1
    if key is None:
      imgs, hdrs = [], []
      for key in sorted(self._dImgs):
        imgs.extend(self._dImgs[key])
        hdrs.extend(self._dHdrs[key])
      return combineImgs(imgs, hdrs=hdrs, nc=self._dNc)[0] if imgs else None

    if c is not None:
      c[key] += 1

    assert i1 is not None and i2 is not None
    if key not in self._dImgs:
      self._dImgs[key], self._dHdrs[key] = [], []
    imgs, hdrs = self._dImgs[key], self._dHdrs[key]
    n = len(imgs)/(i2-i1)
    if n > 1:
      return
    trx = self.trx[0]
    imgs.extend(trx._annImgs(i1, i2, show='td'))
    for i in range(i1, i2):
      if i == i1:
        hdr = "frame %d" %i
      else:
        hdr = key if i == i1+1 and n == 0 else ""
      hdrs.append(hdr)

  def _delayCheckError(self, msg, i1, i2, data, expl=''):
    self._delayImg(i1, i2, msg)
    cv2.imwrite(DELAY_IMG_FILE, self._delayImg())
    error('\n%s %s%s' %(msg, data, '\n'+expl if expl else ''))

  def delayCheck(self):
    print '\n"LED on" delay check'
    trx = self.trx[0]   # fly 1
    ts = trx.ts
    if ts is None:
      print "  skipped (timestamps missing)"
      return
    self.initLedDetector()

    print '  reading frames around each "LED on" event...'
    kLd, kM = 'long delay', 'measured'
    c, dlts, preD, ledMax, npr = collections.Counter(), [], 2, 0, 0
    ldfs = [[] for t in self.trns]
    for i, fi in enumerate(self.on):
      npr += 1   # events processed
      printF('\r  %d: %d' %(i, fi))
      t = Training.get(self.trns, fi)
      if not t:
        c['not training (wake-up)'] += 1
        continue
      f1, f2 = fi-preD, fi+3
      cx, cy, r = t.circles()[False]
      isIn = [distance(trx.xy(j), (cx, cy)) < r for j in range(f1, f2)]
      en = np.nonzero(np.diff(np.array(isIn, np.int)) == 1)[0]
      if en.size != 1:
        self._delayImg(f1, f2, '%d enter events' %en.size, c)
        continue
      ts1, en = ts[f1:f2], en[0]+1
      if np.any(np.diff(ts1) > 1.5/self.fps):
        self._delayImg(f1, f2, 'missing frame', c)
        continue
      vs = [self.feature(readFrame(self.cap, j)) for j in range(f1+en, f2)]
      ledMax = max(ledMax, max(vs))
      isOn = [v > self.ledTh for v in vs]
      if isOn[0]:
        self._delayImg(f1, f2, 'not off at enter', c)
        continue
      if np.any(trx.nan[f1:f1+en+1]):
        self._delayImg(f1, f2, 'fly lost', c)
        continue
      on = np.nonzero(isOn)[0]
      if not on.size:
        expl = '  "on" hard to detect for HtL corner/side chambers, ' + \
          'possibly adjust --dlyCk' if self.ct is CT.htl else ''
        self._delayCheckError('missing "on"', f1, f2, (isIn, en, isOn), expl)
      else:
        dlt = ts1[on[0]+en] - ts1[en]
        c[kM] += 1
        if dlt < .5/self.fps:
          self._delayCheckError('"on" too soon', f1, f2, (isIn, en, isOn))
        if dlt > 1.5/self.fps:
          self._delayImg(f1, f2, kLd, c)
          ldfs[t.n-1].append(fi)
        dlts.append(dlt)

    tc = sum(c[k] for k in c if k not in (kLd, kM))
    assert tc + c[kM] == npr
    print '\n  skipped "LED on" events:%s' %(
      " ({:.1%})".format(tc/npr) if tc else "")
    if tc:
      for k in sorted(c):
        if k != kM:
          print "    %d (%s): %s%s" %(c[k], "{:.1%}".format(c[k]/npr), k,
            " (not skipped)" if k == kLd else "")
    else:
      print "    none"
    print '  classifier: avg. off: %.1f, threshold: %.1f, max. on: %.1f' %(
      self.ledOff, self.ledTh, ledMax)
    print '  "LED on" events measured: %d' %c[kM]
    if c[kM]:
      print '    delay: mean: %.3fs, min: %.3fs, max: %.3fs  (1/fps: %.3fs)' %(
        np.mean(dlts), np.amin(dlts), np.amax(dlts), 1/self.fps)
      if c[kLd]:
        print '    long delays (> 1.5/fps): {:.1%}'.format(c[kLd]/c[kM])
        for i, t in enumerate(self.trns):
          if ldfs[i]:
            print '      t%d: %s' %(t.n, join(", ", ldfs[i], lim=8, end=True))
    img = self._delayImg()
    if img is not None:
      cv2.imwrite(DELAY_IMG_FILE, img)

# - - -
# TODO: class PostAnalysis or AllVideoAnalysis?

# returns t, p, na, nb
def ttest_rel(a, b, msg=None, min_n=2): return ttest(a, b, True, msg, min_n)
def ttest_ind(a, b, msg=None, min_n=2): return ttest(a, b, False, msg, min_n)

def ttest(a, b, paired, msg=None, min_n=2):
  if paired:
    abFinite = np.isfinite(a) & np.isfinite(b)
  a, b = (x[abFinite if paired else np.isfinite(x)] for x in (a, b))
  na, nb = len(a), len(b)
  if min(na, nb) < min_n:
    return np.nan, np.nan, na, nb
  with np.errstate(all='ignore'):
    t, p = st.ttest_rel(a, b) if paired else st.ttest_ind(a, b)
  if msg:
    print "%spaired t-test -- %s:" %("" if paired else "un", msg)
    print "  n = %s means: %.3g, %.3g; t-test: p = %.5f, t = %.3f" %(
      "%d," %na if paired else "%d, %d;" %(na, nb),
      np.mean(a), np.mean(b), p, t)
  return t, p, na, nb

# returns t, p, na
def ttest_1samp(a, val, msg=None, min_n=2):
  a = a[np.isfinite(a)]
  na = len(a)
  if na < min_n:
    return np.nan, np.nan, na
  with np.errstate(all='ignore'):
    t, p = st.ttest_1samp(a, val)
  if msg:
    print "one-sample t-test -- %s:" %msg
    print "  n = %d, mean: %.3g, value: %.1g; t-test: p = %.5f, t = %.3f" %(
      na, np.mean(a), val, p, t)
  return t, p, na

# calculate AUC for each row, returning NaN for rows with missing values
def areaUnderCurve(a):
  if np.all(np.isnan(a[:,-1])):
    a = a[:,:-1]
  assert np.isnan(np.trapz([1,np.nan]))
  return np.trapz(a, axis=1)

# write image or plot
def writeImage(fn, img=None):
  print "writing %s..." %fn
  if img is None:
    plt.savefig(fn, bbox_inches='tight')
  else:
    cv2.imwrite(fn, img)

def headerForType(va, tp, calc):
  if tp in ('atb', 'adb'):
    return "\naverage %s between %s rewards:" %(
      "time" if tp == 'atb' else "distance traveled", cVsA_l(calc))
  elif tp in ('nr', 'nrc'):
    return "\nnumber %s rewards by sync bucket:" %cVsA_l(calc, tp == 'nrc')
  elif tp == 'ppi':
    return "\npositional PI (r*%s) by post bucket:" \
      %formatFloat(opts.radiusMult, 2)
  elif tp == 'rpi':
    return "\n%s reward PI by sync bucket:" %cVsA_l(True)
  elif tp == 'rpip':
    return ""
  elif tp == 'nrp':
    return "\nnumber %s rewards by post bucket:" %cVsA_l(True)
  elif tp == 'nrpp':
    return ""
  elif tp == 'c_pi':
    if va.openLoop:
      return "\npositional preference for LED side:"
    else:
      h = "positional preference (for top)"
      h1 = "\n" + skipMsg() if opts.skip else ""
      return "\n" + ("%s by bucket:" %h if opts.numBuckets > 1 else
        '"%s, including %s-min post bucket:"' %(h, bucketLenForType(tp)[1])
        ) + h1
  elif tp == 'rdp':
    return "\naverage RDP line length (epsilon %.1f)" %opts.rdp
  elif tp == 'bysb2':
    return va.bySB2Header if hasattr(va, 'bySB2Header') else None
  elif tp == 'frc':
    return "\nfirst reward in first sync bucket is control:"
  elif tp == 'xmb':
    return "\ncrossed midline before first reward in first sync bucket:"
  elif tp == 'spd':
    return "\naverage %s:" %va.speedLbl
  elif tp == 'stp':
    return "\naverage stop fraction:"
  elif tp == 'rpm':
    return "\nrewards per minute:"
  else:
    raise ArgumentError(tp)

def fliesForType(va, tp, calc=None):
  if tp in ('atb', 'nr', 'nrc'):
    return va.flies if calc else (0,)
  elif tp in ('ppi', 'frc', 'xmb', 'rpm'):
    return (0,)
  elif tp in ('adb', 'nrp', 'nrpp', 'rpi', 'rpip', 'c_pi', 'rdp', 'bysb2',
      'spd', 'stp'):
    return va.flies
  else:
    raise ArgumentError(tp)

# returns minutes as float and formatted
def bucketLenForType(tp):
  bl = None
  if tp in ('nr', 'nrc', 'rpi', 'bysb2'):
    bl = opts.syncBucketLenMin
  elif tp in ('ppi', 'c_pi'):
    bl = opts.piBucketLenMin
  elif tp in ('nrp', 'nrpp'):
    bl = opts.postBucketLenMin
  elif tp == 'rpip':
    bl = opts.rpiPostBucketLenMin
  return bl, bl if bl is None else formatFloat(bl, 1)

def columnNamesForType(va, tp, calc, n):
  def fiNe(pst, f=None):
    if va.noyc and f == 1: return ()
    fly = "" if f is None else "fly %d " %(f+1)
    return "%sfirst%s" %(fly, pst), "%snext%s" %(fly, pst)
  bl = bucketLenForType(tp)[1]
  if tp in ('atb', 'adb'):
    nr = " %d" %n
    return fiNe(nr, 0) + fiNe(nr, 1) if calc or tp == 'adb' else fiNe(nr)
  elif tp in ('nr', 'nrc'):
    bl = " %s min" %bl
    return fiNe(bl, 0) + fiNe(bl, 1) if calc else fiNe(bl)
  elif tp == 'ppi':
    return ("post %s min" %bl,)
  elif tp in ('rpi', 'bysb2'):
    n = len(vaVarForType(va, tp, calc)[0])
    bl = "%s min" %bl
    def cols(f):
      if va.noyc and f == 1: return ()
      cs =["#%d" %(i+1) for i in range(n)]
      cs[0] = "fly %d %s %s" %(f+1, bl, cs[0])
      return tuple(cs)
    return cols(0) + cols(1)
  elif tp == 'nrp':
    bl = " %s min" %bl
    def cols(f):
      if va.noyc and f == 1: return ()
      cs = ("trn. last", "post 1st", "post 2nd", "post 3rd")
      return tuple("fly %d %s%s" %(f+1, c, bl) for c in cs)
    return cols(0) + cols(1)
  elif tp in ('nrpp', 'rpip'):
    return None
  elif tp == 'c_pi':
    if va.openLoop:
      ps = (" pre",) + (("",) if va.alt else (" on", " off"))
    else:
      ps = ("", " post") if opts.numBuckets == 1 else (" first", " next")
    def cols(f):
      if va.noyc and f == 1: return ()
      return tuple("fly %d%s" %(f+1, p) for p in ps)
    return cols(0) + cols(1)
  elif tp == 'rdp':
    return "fly 1", "fly 2"
  elif tp in ('frc', 'xmb', 'rpm'):
    return ("fly 1",)
  elif tp in ('spd', 'stp'):
    def cols(f):
      if va.noyc and f == 1: return ()
      f = "fly %d " %(f+1)
      return (f+"pre", f+"training")
    return cols(0) + cols(1)
  else:
    raise ArgumentError(tp)

def vaVarForType(va, tp, calc):
  if tp == 'atb': return va.avgTimeBtwnCalc if calc else va.avgTimeBetween
  elif tp == 'adb': return va.avgDistBtwnCalc if calc else va.avgDistBetween
  elif tp in ('nr', 'nrc'): return va.numRewards[calc][tp == 'nrc']
  elif tp == 'ppi': return va.posPI
  elif tp == 'rpi': return va.rewardPI
  elif tp == 'rpip': return va.rewardPiPst
  elif tp == 'nrp': return va.numRewardsPost
  elif tp == 'nrpp': return va.numRewardsPostPlot
  elif tp == 'c_pi': return va.posPI
  elif tp == 'rdp': return va.rdpAvgLL
  elif tp == 'bysb2': return va.bySB2
  elif tp == 'frc': return va.firstRewardCtrl
  elif tp == 'xmb': return va.xedMidlineBefore
  elif tp == 'spd': return va.speed
  elif tp == 'stp': return va.stopFrac
  elif tp == 'rpm': return va.rewardsPerMin
  else:
     raise ArgumentError(tp)

def trnsForType(va, tp):
  if tp == 'ppi': return [] if opts.rdp else va.posPITrns
  elif tp == 'rpi': return va.rewardPITrns
  elif tp == 'rdp': return va.trns[-1:] if opts.rdp else []
  else: return va.trns

def typeCalc(tc):
  ps = tc.split('-')
  return ps[0], ps[1] == 'c' if len(ps) > 1 else False

# make sure values for bad trajectories are NaN
def checkValues(vas, tp, calc, a):
  fs = fliesForType(vas[0], tp, calc)
  npf = int(a.shape[2]/len(fs))
  for i, va in enumerate(vas):
    for f in fs:
      if va._bad(f):
        assert np.all(np.isnan(a[i,:,f*npf:(f+1)*npf]))

FLY_COLS = ('#1f4da1', '#a00000')

# plot reward PIs or rewards post training
# a: data; gis: array with group index for each video
# TODO: shorten this function
def plotRewards(va, tp, a, trns, gis, gls, vas=None):
  nrp, rpip = tp == 'nrpp', tp == 'rpip'
  post = nrp or rpip
  nnpb = va.rpiNumNonPostBuckets if rpip else va.numNonPostBuckets
  fs, ng = fliesForType(va, tp), gis.max()+1
  nf = len(fs)
  nb, (meanC, fly2C) = int(a.shape[2]/nf), FLY_COLS
  meanOnly, showN, showV, joinF, fillBtw = True, True, False, True, True
  showPG, showPP = True, True    # p values between groups, for post
  showPFL = True                 # p values between first and last buckets
  showPT = not P                 # p values between trainings
  showSS = not P                 # speed stats
  if showSS and vas:
    speed, stpFr = (np.array([getattr(va, k) for va in vas]) for k in (
      'speed', 'stopFrac'))
    speed, stpFr = (np.nanmean(a, axis=0) for a in (speed, stpFr))
  nr = 1 if joinF else nf
  bl, blf = bucketLenForType(tp)
  xs = (np.arange(nb) + (-(nnpb-1) if post else 1))*bl
  ylim = [0, 60] if nrp else [-1, 1]
  lbls, fbv = {}, []
  tas = 2*[None]   # index: 0:under curve, 1:between curves
  if P and F2T: trns = trns[:2]
  nc = len(trns)
  axs = plt.subplots(nr, nc,
    figsize=pch(([5.33, 11.74, 18.18][nc-1], 4.68*nr), (20, 5*nr)))[1]
  if nr == 1:
    if nc == 1: axs = np.array([[axs]])
    else: axs = axs[None]
  for f in fs:
    mc = fly2C if joinF and f == 1 else meanC
    for i, t in enumerate(trns):
      nosym = not t.hasSymCtrl()
      comparable = not (nf == 1 and nosym)
      ax = axs[0 if joinF else f, i]
      plt.sca(ax)
      if P and f == 0:
        plt.locator_params(axis='y', nbins=5)
      # delta: return difference between fly 0 and fly 1
      def getVals(g, b=None, delta=False, f1=None):
        vis = np.flatnonzero(gis == g)
        def gvs(f):
          o = f*nb
          return a[vis,i,o:o+nb] if b is None else a[vis,i,o+b]
        return gvs(0)-gvs(1) if delta else gvs(f1 if f1 is not None else f)
      if not meanOnly:
        # plot line for each video
        assert ng == 1
        for v in range(a.shape[0]):
          ys = a[v,i,f*nb:(f+1)*nb]
          fin = np.isfinite(ys)
          plt.plot(xs[fin], ys[fin], color='0.7', marker='o', ms=3)
      # plot mean and confidence interval
      for g in range(ng):   # group
        mci = np.array([meanConfInt(getVals(g, b)) for b in range(nb)]).T
          # 4 rows: mean, lower bound, upper bound, number samples
        if not (rpip and f == 1 and not nosym):
          for j in range(3):
            ys = mci[j,:]
            fin = np.isfinite(ys)
            if j == 0 or not fillBtw:
              line, = plt.plot(xs[fin], ys[fin], color=mc,
                marker='o', ms=3 if j == 0 else 2, mec=mc,
                linewidth=2 if j == 0 else 1,
                linestyle='-' if j == 0 and g == 0 else '--')
              if i == 0 and j == 0 and f == 0 and gls:
                line.set_label(gls[g] + (' yoked-ctrl' if f else ''))
            if j == 2 and fillBtw:
              plt.fill_between(xs[fin], mci[1,:][fin], ys[fin], color=mc,
                alpha=.15)
          # sample sizes
          if showN and (not nrp or i == 0) and (ng == 1 or f == 0):
            for j, n in enumerate(mci[3,:1] if nrp else mci[3,:]):
              if n > 0:
                y, key, m = mci[0,j], join('|', (i,j)), (ylim[1]-ylim[0])/2
                txt = pltText(xs[j], y+.04*m, "%d" %n,
                  ha='center', size=pch(11, 'x-small'), color='.2')
                txt1 = lbls.get(key)
                if txt1:
                  y1 = txt1._y_
                  txt1._firstSm_ = y1 < y
                  if abs(y1-y) < pch(.14, .1)*m:   # move label below
                    txta, ya = (txt, y) if y1 > y else (txt1, y1)
                    txta.set_y(ya-pch(.04, .03)*m)
                    txta.set_va('top')
                    txta._ontp_ = False
                else:
                  txt._y_, txt._ontp_, txt._firstSm_ = y, True, False
                  lbls[key] = txt
          # values
          if showV:
            for j, y in enumerate(mci[0,:]):
              if np.isfinite(y):
                pltText(xs[j], y-.08*(30 if nrp else 1),
                  ("%%.%df" %(1 if nrp else 2)) %y,
                  ha='center', size='xx-small', color='.2')
        # t-test p values
        if (showPG and ng == 2 and g == 1 and f == 0 or
            rpip and showPP and ng == 1 and f == nf-1 and comparable) \
            and not nrp:
          cmpg, dlt = g == 1, nosym if nf == 2 else False
          tpm = np.array([
            (ttest_ind(getVals(0, b, dlt), getVals(1, b, dlt)) if cmpg else
              ttest_1samp(getVals(0, b, nosym, 0), 0))[:2] +
            (np.nanmean(getVals(int(cmpg), b)),) for b in range(nb)]).T
            # 3 rows: t-test t and p and mean for g == int(cmpg)
          assert isClose(mci[0,:], tpm[2,:])
          for j, p in enumerate(tpm[1,:]):
            txt = lbls.get(join('|', (i,j)))
            if txt:
              y, ontp, fs = txt._y_, txt._ontp_, txt._firstSm_
              strs = p2stars(p, nanR='')
              sws = strs.startswith("*")
              if not cmpg and not nosym and not sws:
                continue
              y += 0 if sws else pch(.02, .015)*m
              ys = y-pch(.15, .105)*m if not ontp else (
                y-pch(.06, .045)*m if fs else y+pch(.13, .1)*m)
              pltText(xs[j], ys, strs,
                ha='center', va=('baseline' if ys > y else 'top'),
                size=pch(11, 'x-small'), color='0', weight='bold')
          # AUC
          if not rpip:
            if i == 0:
              print "\narea under reward index curve or between curves " + \
                "by group:"
            yp = -0.79 if nosym else pch(-.55, -.46)
            for btwn in pch((False,), (False, True)):
              if nosym and not btwn or nf == 1 and btwn:
                continue
              a_ = tuple(areaUnderCurve(getVals(x, None, btwn)) for x in (0, 1))
              if tas[btwn] is None:
                tas[btwn] = a_
              else:
                tas[btwn] = tupleAdd(tas[btwn], a_)
              for tot in (False, True):
                if i == 0 and tot:
                  continue
                def getA(g):
                  return (tas[0][g] + a_[g] if nosym else tas[btwn][g]) \
                    if tot else a_[g]
                try:
                  a0, a1 = getA(0), getA(1)
                except TypeError:   # triggered, e.g., for 3x center training
                  continue
                nm = pcap(("total " if tot else "") + ("AUC + ABC"
                  if nosym and tot else ("ABC" if btwn else "AUC")))
                tpn = ttest_ind(a0, a1, "%s, %s" %(
                  "training 1-%d" %(i+1) if tot else t.name(), nm))
                pltText(xs[0], yp,
                  "%s (n=%d,%d): %s" %(
                    nm, tpn[2], tpn[3], p2stars(tpn[1], True)),
                  size=pch(12, 'small'), color='0')
                yp -= pch(.14, .11)
        # t-test first vs. last
        if showPFL and ng == 1 and f == 0 and not post and comparable:
          lb = nb - 1
          while True:
            tpn = ttest_rel(getVals(0, 0, nosym), getVals(0, lb, nosym))
            if tpn[3] < 2 and lb > 1: lb = lb - 1
            else: break
          with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore", r'Mean of empty slice')
            ms = np.array([np.nanmean(getVals(0, b)) for b in range(nb)])
          assert isClose(mci[0,:], ms)
          x1, x2 = xs[0], xs[lb]
          y, h, col = ms[0:lb+1].max() + pch(.15, .13), .03, '0'
          if np.isfinite(y):
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            pltText((x1+x2)*.5, y+h, p2stars(tpn[1]), ha='center', va='bottom',
              size=pch(11, 'small'), color=col, weight='bold')
            if y+h > .9: ylim[1] = y+h+.1
        # t-test between trainings
        if showPT and ng == 1 and f == 0 and not post and comparable:
          assert len(fbv) == i
          fbv.append(getVals(0, 0, nosym))
          if i > 0 and t.hasSymCtrl() == trns[0].hasSymCtrl():
            tpn = ttest_rel(fbv[0], fbv[i])
            pltText(xs[0], -.7,
              "1st bucket, t1 vs. t%d (n=%d): %s" %(
                i+1, min(tpn[2], tpn[3]), p2stars(tpn[1], True)),
              size='small', color='0')
        # speed stats
        if showSS and ng == 1 and f == 1 and not post:
          for f1 in va.flies:
            i1 = i*2+f1
            pltText(xs[0], -.83-f1*.11,
              "f%s: %s/s: %s, stop: %s" %(f1+1,
              "mm" if SPEED_ON_BOTTOM else "px",
              join(", ", speed[i1], p=1), join(", ", stpFr[i1], p=2)),
              size='small', color='0')
      # labels etc.
      if f == 0 or not joinF:
        plt.title(pcap(("post " if post else "") + (t.name() if joinF else
          (("%s " %t.name() if f == 0 else "") + "fly %d" %(f+1)))))
        plt.xlabel(pcap("end points [min] of %s min %sbuckets" %(
          blf, "" if post and not (rpip and POST_SYNC is not ST.fixed) else
            "sync ")))
        if not P or i == 0:
          plt.ylabel(pcap("circle enter events" if nrp else "reward index"))
        plt.axhline(color='k')
        if post:
          plt.xlim(xs[0]-bl, xs[-1]+bl)
          plt.ylim(*ylim)
          if nnpb > 0:   # "training end" line
            xl = xs[nnpb-1]
            plt.plot([xl, xl], ylim, color='0.5', linewidth=2, linestyle='--',
              zorder=1)
        else:
          plt.xlim(0, xs[-1])
          plt.ylim(*ylim)
      if i == 0 and joinF and ng == 1 and nf == 2 and not P:
        pltText(.85, (0.87 if nrp else 0.18)-f*.08, 'fly %d' %(f+1),
          color=mc, transform=ax.transAxes)
      if i == 0 and f == 0 and gls and (not P or LEG):
        plt.legend(loc=1 if nrp else 4,
          prop=dict(size='medium', style='italic'))
  if not nrp:
    plt.subplots_adjust(wspace=opts.wspace)
  writeImage((REWARDS_IMG_FILE if nrp else
    (REWARD_PI_POST_IMG_FILE if rpip else REWARD_PI_IMG_FILE)) %blf)

# plot turn angles and run lengths
def plotRdpStats(vas, gls, tpTa=True):
  if tpTa:
    binW = 10
    bins = np.arange(-180,180.1,binW)
    cntr, barW, barO = (bins[:-1] + bins[1:]) / 2, 0.35*binW, 0.4*binW
  else:
    cntr, barW, barO = np.array([0]), 0.35, 0.4
  nb, nf, flies = len(cntr), [], vas[0].flies
  plt.figure(figsize=(15 if tpTa else 4, 6))
  for f in flies:
    if gls and f == 1:
      continue
    for g in range(len(gls) if gls else 1):   # group
      byFly = []
      for va in vas:
        if gls and va.gidx != g:
          continue
        if tpTa:
          ta = va.rdpTA[f]
          rdpTA = np.concatenate(ta if ta else [[]])*180/np.pi
          if len(rdpTA) >= RDP_MIN_TURNS:
            hist, edg = np.histogram(rdpTA, bins=bins, density=True)
            byFly.append(hist * binW)
        else:
          mll = va.rdpAvgLL[f]
          if not np.isnan(mll):
            byFly.append(mll)
      nf.append(len(byFly))
      byFly = np.array(byFly)
      mci = np.array([meanConfInt(byFly[:,b]) for b in range(nb)]).T
        # 4 rows: see plotRewards()
      assert isClose(mci[0,:], np.mean(byFly, axis=0))
      bars = plt.bar(cntr + barO*(f+g-.5), mci[0], align='center', width=barW,
        color=FLY_COLS[f], edgecolor=FLY_COLS[f], linewidth=1,
        fill = False if g else True,
        yerr=[mci[0]-mci[1], mci[2]-mci[0]], ecolor='.6', capsize=0,
          error_kw=dict(elinewidth=2))
      if gls:
        bars.set_label(gls[g])

  # labels etc.
  plt.title(va.rdpInterval)
  plt.xlabel("turn angle [degrees]" if tpTa else "")
  plt.ylabel("relative frequency" if tpTa else "average run length [pixels]")
  if not tpTa:
    plt.xlim(-2, 2)
    plt.ylim(0, plt.ylim()[1]*1.2)
    plt.xticks([])
  tf = plt.gca().transAxes
  if gls:
    plt.legend(loc=1, prop=dict(size='medium', style='italic'))
    plt.text(0.9 if tpTa else 0.72, 0.75, 'n=%d,%d' %tuple(nf),
      size='small', color='.2', transform=tf)
  else:
    for f in flies:
      yt = (0.85 if tpTa else 0.9)-f*.06
      plt.text(0.86 if tpTa else 0.6, yt, 'fly %d' %(f+1),
        color=FLY_COLS[f], transform=tf)
      plt.text(0.915 if tpTa else 0.8, yt, 'n=%d' %nf[f], size='small',
        color='.2', transform=tf)

  writeImage(TURN_ANGLES_IMG_FILE if tpTa else RUN_LENGTHS_IMG_FILE)

# plot heatmaps
def plotHeatmaps(vas):
  if max(va.gidx for va in vas) > 0:
    return
  prob = True   # show probabilities (preferred)
  cmap = mplColormap()   # alternatives: inferno, gray, etc.
  usesb = False   # Seaborn heatmaps have lines for alpha < 1
  va0, alpha = vas[0], 1 if opts.bg is None else opts.bg
  trns, lin, flies = va0.trns, opts.hm == OP_LIN, va0.flies
  if P and F2T: trns = trns[:2]
  imgs, nc, nsc = [], len(trns), 2 if va0.ct is CT.regular else 1
  nsr, nf = 1 if va0.noyc else 3 - nsc, len(flies)
  if va0.ct is CT.regular:
    fig = plt.figure(figsize=(4*nc,6))
  elif va0.ct is CT.large:
    fig = plt.figure(figsize=(3.1*nc,6*nsr))
  gs = mpl.gridspec.GridSpec(2, nc+1, wspace=.2, hspace=.2/nsr,
    width_ratios=[1]*nc+[.07], top=.9, bottom=.05, left=.05, right=.95)
  cbar_ax = []
  for pst in (0, 1):
    def hm(va): return va.heatmapPost if pst else va.heatmap
    cbar_ax.append(fig.add_subplot(gs[pst,nc]))
    mpms, nfs, vmins = [], [], []
    for i, f in itertools.product(range(nc), flies):
      mps, ls = [], []
      for va in vas:
        mp, l = hm(va)[f][i][:2]
        if mp is not None and np.sum(mp) > 0:
          mps.append(mp/l if prob else mp)
          ls.append(l)
      assert np.all(np.abs(np.diff(ls)) <= 2)   # about equal numbers of frames
      mpm = np.mean(mps, axis=0)
      mpms.append(mpm)
      nfs.append(len(mps))
      vmins.append(np.amin(mpm[mpm>0]))
    vmin, vmax = np.amin(vmins), np.amax(mpms)
    vmin1 = 0 if lin else vmin/(vmax/vmin)**.05   # .9*vmin not bad either
    for i, t in enumerate(trns):
      imgs1 = []
      gs1 = mpl.gridspec.GridSpecFromSubplotSpec(nsr, nsc,
        subplot_spec=gs[pst,i],
        wspace=.06 if nsc>1 else 0., hspace=.045 if nsr>1 else 0.)
      ttl = pcap("post %s min%s" %(formatFloat(opts.rpiPostBucketLenMin, 1),
        "" if POST_SYNC is ST.fixed else " sync") if pst else t.name())
      for f in flies:
        mp = mpms[i*nf+f]
        mp = np.maximum(mp, vmin1)
        if f == 0:
          ttln = "n=%d" %nfs[i*nf+f]
        img = cv2.resize(heatmap(mp, xform=None if lin else np.log),
          (0,0), fx=HEATMAP_DIV, fy=HEATMAP_DIV)
        ax = fig.add_subplot(gs1[f])
        if usesb:
          sns.heatmap(mp, ax=ax, alpha=alpha,
            square=True, xticklabels=False, yticklabels=False,
            cmap=cmap, vmax=vmax, vmin=vmin1,
            norm=None if lin else mpl.colors.LogNorm(),
            cbar=i==0 and f==0,
            cbar_kws=None if lin else dict(
              ticks=mpl.ticker.LogLocator(subs=(1.,3.)),
              format=mpl.ticker.LogFormatter(minor_thresholds=(10,10))),
            cbar_ax=None if i or f else cbar_ax[pst],
          )
        else:
          ai = ax.imshow(mp, alpha=alpha,
            cmap=cmap, vmax=vmax, vmin=vmin1,
            norm=None if lin else mpl.colors.LogNorm(),
            extent=[0, mp.shape[1], mp.shape[0], 0],
          )
          ax.set(xticks=[], yticks=[], aspect="equal")
          ax.axis("off")
          if i == 0 and f == 0:
            kws = {} if lin else dict(
              ticks=mpl.ticker.LogLocator(subs=(1.,3.)),
              format=mpl.ticker.LogFormatter(minor_thresholds=(10,10)))
            cb = ax.figure.colorbar(ai, cbar_ax[pst], ax, **kws)
            cb.outline.set_linewidth(0)
            cb.solids.set_alpha(1)
            cb.solids.set_cmap(alphaBlend(cmap, alpha))
        xym = hm(va0)[f][i][2]
        if opts.bg is not None:   # add chamber background
          wh = tupleMul(mp.shape[::-1], HEATMAP_DIV)
          tl, br = (va0.xf.t2f(*xy) for xy in (xym, tupleAdd(xym, wh)))
          ax.imshow(va0.background()[tl[1]:br[1], tl[0]:br[0]],
            extent=ax.get_xlim() + ax.get_ylim(),
            cmap='gray', vmin=0, vmax=255, zorder=-1)
        if f == 0:
          plt.title(ttl, loc='left')
        if (f == 0) == (nsc == 1):
          plt.title(ttln, loc='right', size='medium')
        if not pst and f == 0:
          cx, cy, r = t.circles(f)[0]
          cxy = tupleSub(va0.mirror(va0.xf.f2t(cx, cy)), xym)
          cv2.circle(img, intR(cxy), r, COL_W if lin else COL_BK, 1)
          ax.add_artist(mpl.patches.Circle(tupleMul(cxy, 1./HEATMAP_DIV),
            r/HEATMAP_DIV, color='w' if lin else 'k', fill=False,
            linewidth=.8))
        imgs1.append(img)
      imgs.append((combineImgs(imgs1, nc=nsc, d=5)[0], ttl + " (%s)" %ttln))
  img = combineImgs(imgs, nc=nc)[0]
  writeImage(HEATMAPS_IMG_FILE %"", img)
  writeImage(HEATMAPS_IMG_FILE %2)
  oob = [basename(va.fn) for va in vas if va.heatmapOOB]
  if oob:
    warn("heatmaps out of bounds for %s" %commaAndJoin(oob))
  if False:   # for showing mean distance
    for f in flies:
      print ">>> fly %d: %.3g" %(f+1,
        np.mean([va.trx[f].mean_d for va in vas if not va.trx[f].bad()]))

# "post analyze" the given VideoAnalysis objects
def postAnalyze(vas):
  if len(vas) <= 1:
    return

  print "\n\n=== all video analysis (%d videos) ===" %len(vas)
  print "\ntotal rewards training: %d" %sum(
    va.totalTrainingNOn for va in vas)

  n, va = opts.numRewardsCompare, vas[0]
  gis = np.array([va.gidx for va in vas])
  gls = opts.groupLabels and opts.groupLabels.split('|')
  ng = gis.max()+1
  if gls and len(gls) != ng:
    error('numbers of groups and group labels differ')

  if not (va.circle or va.choice):
    return

  tcs = ('bysb2',) if va.choice else (
    'atb', 'atb-c', 'adb', 'adb-c', 'nr', 'nr-c', 'ppi',
    'rpi', 'rpip', 'nrp-c', 'nrpp-c', 'rdp', 'bysb2', 'spd', 'stp', 'rpm')
  for tc in tcs:
    tp, calc = typeCalc(tc)
    hdr = headerForType(va, tp, calc)
    if hdr is None: continue
    print hdr
    cns = columnNamesForType(va, tp, calc, n)
    nf = len(fliesForType(va, tp, calc))
    if cns:
      nb = int(len(cns)/nf)
    trns = trnsForType(va, tp)
    if not trns:
      print "skipped"
      continue
    a = np.array([vaVarForType(va, tp, calc) for va in vas])
    a = a.reshape((len(vas), len(trns), -1))
    # a's dimensions: video, training, bucket or fly x bucket
    assert cns is None or a.shape[2] == len(cns)
    checkValues(vas, tp, calc, a)
    if tp == 'ppi':
      for i, t in enumerate(trns):
         ttest_1samp(a[:,i,0], 0, "%s %s" %(t.name(), cns[0]))
    elif tp == 'rpi':
      for i, t in enumerate(trns):
        if t.hasSymCtrl():
          for j, cn in enumerate(cns):
            ttest_1samp(a[:,i,j], 0, "%s %s" %(t.name(), cn))
      plotRewards(va, tp, a, trns, gis, gls, vas)
      if len(trns) > 1 and all(t.hasSymCtrl() for t in trns[:2]):
        ttest_rel(a[:,0,0], a[:,1,0], "first sync bucket, training 1 vs. 2")
      for i, t in enumerate(trns):
        if nf == 1 and not t.hasSymCtrl(): continue
        lb = nb - 1
        while True:
          ab = [a[:,i,b] if t.hasSymCtrl() else a[:,i,b]-a[:,i,b+nb]
            for b in (0, lb)]
          nbt = ttest_rel(ab[0], ab[1], "%s, fly %s, bucket #%d vs. #%d" %(
            t.name(), "1" if t.hasSymCtrl() else "delta", 1, lb+1))[3]
          if nbt < 2 and lb > 1: lb = lb - 1
          else: break
    elif tp == 'rpip':
      plotRewards(va, tp, a, trns, gis, gls)
    elif tp == 'nrp':
      for i, t in enumerate(trns):
        for i1, i2 in ((0,1), (4,5), (0,4), (1,5), (2,6), (3,7)):
          if i2 < a.shape[2]:
            ttest_rel(a[:,i,i1], a[:,i,i2], "training %d, %s vs. %s" %(
              t.n, cns[i1], cns[i2]))
    elif tp == 'nrpp':
      plotRewards(va, tp, a, trns, gis, gls)
    elif tp == 'rdp':
      ttest_rel(a[:,0,0], a[:,0,1], va.rdpInterval + ", %s vs. %s" %cns[:2])
      plotRdpStats(vas, gls, False)
    elif tp == 'bysb2':
      for i, t in enumerate(trns):
        ab = [np.hstack((a[:,i,b], a[:,i,b+nb])) if opts.ol else a[:,i,b]
          for b in (0, nb-1)]
        ttest_rel(ab[0], ab[1], "%s, bucket #%d vs. #%d" %(t.name(), 1, nb))
    elif tp in ('spd', 'stp', 'rpm'):
      spst = tp in ('spd', 'stp')
      fm = "{:.1f}" if tp == 'rpm' else ("{:.2f}" if tp == 'spd' else "{:.1%}")
      if ng == 1 and spst and nf == 2:
        for i, t in enumerate(trns):
          ttest_rel(a[:,i,1], a[:,i,3], "training %d, %s vs. %s" %(
            t.n, cns[1], cns[3]))
      print "means with 95%% confidence intervals%s:" %(
        " (pre, training)" if spst else "")
      if tp == 'spd' and va.ct in (CT.htl, CT.large) and SPEED_ON_BOTTOM:
        print "note: sidewall and lid currently included"
      flies, groups = fliesForType(va, tp) if ng == 1 else (0,), range(ng)
      mgll = None if ng == 1 else max(len(g) for g in gls)
      ns = [np.count_nonzero(gis == g) for g in groups]
      print '  n = %s  (in "()" below if different)' %join(", ", ns)
      for i, t in enumerate(trns):
        for f, g in itertools.product(flies, groups):
          txt = []
          for b in range(nb):
            ci = nb*f + b
            mcn = meanConfInt(a[np.flatnonzero(gis == g),i,ci], asDelta=True)
            sn = mcn[2] != ns[g]
            txt.append(("%s ±%s%s" %(fm, fm, " ({})" if sn else "")).
              format(*mcn[:3 if sn else 2]))
          print "  %s %s: %s" %(
            "t%d," %t.n if f == 0 and g == 0 else " "*3,
            "fly %d" %(f+1) if ng == 1 else gls[g].ljust(mgll), ", ".join(txt))
    # handle "type codes" included in postAnalyze() for checkValues()
    elif tp == None:
      pass
    else:
      adba = tp == 'adb' and not calc
      if (calc or adba) and nf == 2:
        assert nb == 2
        for b in range(1 + adba):
          for i, t in enumerate(trns):
            ttest_rel(a[:,i,b], a[:,i,b+nb], "training %d, %s vs. %s" %(
              t.n, cns[b], cns[b+nb]))
        if not adba:
          ttest_rel(a[:,0,2], a[:,0,3], "training 1, %s vs. %s" %cns[2:])
      if not calc:
        ttest_rel(a[:,0,0], a[:,0,1], "training 1, %s vs. %s" %cns[:2])
        if len(trns) > 1:
          ttest_rel(a[:,0,0], a[:,1,0], "%s, training 1 vs. 2" %cns[0])
      if nf == 1 and calc:
        print "skipped"

  if opts.rdp:
    plotRdpStats(vas, gls)

def writeStats(vas, sf):
  print "\nwriting %s..." %STATS_FILE
  writeCommand(sf, csvStyle=True)

  n, va = opts.numRewardsCompare, vas[0]
  for t in va.trns:
    sf.write('"%s"\n' %t.name(short=False))
  if opts.move:
    return
  tcs = ('c_pi', 'bysb2') if va.choice else (
    'atb', 'atb-c', 'adb', 'adb-c', 'nr', 'nr-c', 'nrc-c', 'ppi', 'nrp-c',
    'rdp', 'bysb2', 'frc', 'xmb', 'spd')
  def frm(n): return "{:.3f}".format(n) if isinstance(n, float) else str(n)
  for tc in tcs:
    tp, calc = typeCalc(tc)
    assert tp != 'nrc' or calc == True
    hdr = headerForType(va, tp, calc)
    if hdr is None: continue
    sf.write(hdr + '\n')
    cns = ",".join(columnNamesForType(va, tp, calc, n))
    trns = trnsForType(va, tp)
    if not trns:
      sf.write('skipped\n')
      continue
    sf.write(('video,' if VIDEO_COL else '') +
      ('fly,' if va.f is not None else '') +
      ','.join('%s %s' %(t.name(), cns) for t in trns) + '\n')
    for va in vas:
      sf.write((basename(va.fn)+',' if VIDEO_COL else '') +
        ("%d," %va.f if va.f is not None else '') +
        ','.join(map(frm, concat(vaVarForType(va, tp, calc), True))) + '\n')

  # custom code for trajectory heatmap analysis
  if hasattr(va, 'avgMaxDist'):
    sf.write('\nheatmap analysis (epsilon %.1f; number traj.: %d)\n' %(
      opts.rdp, va.ntrx))
    vs = (
      ('average maximum distance', 'avgMaxDist'),
      ('average absolute first turn angle', 'avgFirstTA'),
      ('average first run length', 'avgFirstRL'))
    ncols, ntrns = len(va.avgMaxDist[0]), len(va.trns)
    cols = 'video,fly,' + ','.join(
      ','.join('t%d b%d' %(t+1, b+1) for b in range(ncols/ntrns))
      for t in range(ntrns))
    for (hdr, vn) in vs:
      sf.write('\n' + hdr + '\n' + cols + '\n')
      for f in va.flies:
        for va in vas:
          r = getattr(va, vn)[f]
          assert len(r) == ncols
          sf.write(basename(va.fn)+',' + str(f+1)+',' +
            ','.join(map(str, r)) + '\n')

def analysisImage(vas):
  backup(ANALYSIS_IMG_FILE)
  imgs = [(va.aimg, basename(va.fn)) for va in vas if va.aimg is not None]
  img = combineImgs(imgs, nc=5)[0]
  writeImage(ANALYSIS_IMG_FILE, img)

_CAM_DATE = re.compile(r'^(c\d+__[\d-]+)')
def openLoopImage(vas):
  imgs = []
  for va in vas:
    bn = basename(va.fn)
    imgs.append((va.olimg, bn if va.ct is CT.regular else
      firstGroup(_CAM_DATE, bn)))
  writeImage(OPEN_LOOP_IMG_FILE, combineImgs(imgs, nc=5)[0])

# - - -

def analyze():
  if P:
    mpl.rcParams.update({'font.size': 12,   # ignore opts.fontSize
      'xtick.direction': 'in', 'ytick.direction': 'in',
      'xtick.top': True, 'ytick.right': True})
  else:
    mpl.rcParams['font.size'] = opts.fontSize
  mpl.rcParams.update({'axes.linewidth': 1, 'lines.dashed_pattern': '3.05, 3.'})
  vgs = opts.video.split('|')
  ng = len(vgs)
  # flies by group
  if opts.fly is None:
    fs = [[None]]
  else:
    fs = [parseIntList(v) for v in opts.fly.split('|')]
  if len(fs) == 1:
    fs = fs*ng
  if len(fs) != ng:
    error("fly numbers required for each group")
  # fn2fs: file name: list with the lists of fly numbers for each group
  def fctry(): return [[]]*ng
  fn2fs, fnf = collections.defaultdict(fctry), []
  for i, vg in enumerate(vgs):
    for v in vg.split(","):
      vFs = v.split(":")
      if len(vFs) == 1:
        fs1 = fs[i]
      else:
        v, fs1 = vFs[0], parseIntList(vFs[1])
      for fn in fileList(v, 'analyze', pattern=AVI_X):
        fn2fs[fn][i] = fs1
        fnf.extend(fn if f is None else "%s:%d" %(fn, f) for f in fs1)
  dups = duplicates(fnf)
  if dups:
    error('duplicate: %s' %dups[0])
  fns = fn2fs.keys()
  if not fns:
    return
  cns = [int(firstGroup(CAM_NUM, basename(fn))) for fn in fns]
  vas, va = [], None
  for i, fn in enumerate([fn for (cn, fn) in sorted(zip(cns, fns))]):
    for gidx in range(ng):
      for f in fn2fs[fn][gidx]:
        if va:
          print
        va = VideoAnalysis(fn, gidx, f)
        if not va.skipped():
          vas.append(va)

  if vas:
    postAnalyze(vas)
    backup(STATS_FILE)
    with open(STATS_FILE, 'w', 1) as sf:
      writeStats(vas, sf)
    if vas[0].circle or vas[0].choice:
      analysisImage(vas)
    if vas[0].circle:
      if opts.fixSeed:
        random.seed(101)
      try:
        random.choice(vas).calcRewardsImgs()
      except util.VideoError:
        print 'some "rewards images" not written due to video error'
    if opts.hm:
      plotHeatmaps(vas)
    if vas[0].openLoop:
      openLoopImage(vas)
  if opts.showPlots or opts.showTrackIssues:
    plt.show(block=False)
    raw_input("\npress Enter to continue...")

# - - -

# self test
def test():
  Trajectory._test()

# - - -

test()
log = not (opts.showPlots or opts.showTrackIssues)
  # note: Tee makes plt.show(block=False) not work
if log:
  backup(LOG_FILE)
with open(LOG_FILE if log else os.devnull, 'w', 1) as lf:
  writeCommand(lf)
  if log:
    sys.stdout = Tee([sys.stdout, lf])
  analyze()


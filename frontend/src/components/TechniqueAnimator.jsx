/**
 * TechniqueAnimator — Pose-keyframe stick figure boxer
 *
 * ANIMATION PRINCIPLES (direct from @firytwig tutorial):
 *
 * 1. CHAMBER → SNAP → OVERSHOOT → RETRACT
 *    Not: smooth lerp from guard to attack.
 *    The figure HOLDS near guard, then snaps at the last moment.
 *
 * 2. easeInQuart (t^4) for attack phase:
 *    At 50% through attack window → only 6.25% of the motion has happened.
 *    At 90% through attack window → 65% of the motion happens.
 *    Effect: figure looks almost identical to chamber until the last instant,
 *    then snaps to full extension. THIS IS THE SNAP.
 *    "Keep inbetween frames close to the chamber." — @firytwig
 *
 * 3. DO NOT ease-out into the attack pose.
 *    No deceleration. The fist arrives at full speed, then overshoots.
 *
 * 4. OVERSHOOT: extend 6-8 units past the target, then ease back.
 *    The reaction force — "every action has equal and opposite reaction."
 *
 * 5. Full body commitment: every joint moves.
 *    "Put the whole body into the attack — especially the hips."
 *
 * POSE COORDINATE SYSTEM:
 *   Origin = pelvis center
 *   x+ = right (toward opponent)   y+ = down
 *   At scale s (= canvas height / 400), 1 unit = s pixels
 */

import { useEffect, useRef, useState } from 'react'
import styles from './TechniqueAnimator.module.css'

// ─────────────────────────────────────────────────────────────
// EASING FUNCTIONS
// ─────────────────────────────────────────────────────────────
const E = {
  linear:         t => t,
  io2:            t => t < .5 ? 2*t*t : 1-(-2*t+2)**2/2,       // ease-in-out quad
  io3:            t => t < .5 ? 4*t**3 : 1-(-2*t+2)**3/2,      // ease-in-out cubic
  out2:           t => 1-(1-t)**2,
  out3:           t => 1-(1-t)**3,
  out4:           t => 1-(1-t)**4,
  in4:            t => t**4,     // ← THE SNAP EASE. stays near start, snaps at end
  in5:            t => t**5,     // even snappier
}

// ─────────────────────────────────────────────────────────────
// POSE DEFINITIONS
// [dx, dy] relative to pelvis (0,0) in scale units
// Orthodox stance: fighter faces right, left = lead, right = rear
// ─────────────────────────────────────────────────────────────

const GUARD = {
  head:   [  0,  -98],  neck:   [  0,  -80],
  lSh:    [-24,  -68],  rSh:    [ 22,  -68],
  lEl:    [-20,  -42],  rEl:    [ 16,  -44],
  lHand:  [-10,  -26],  rHand:  [  8,  -28],
  lHip:   [-14,    0],  rHip:   [ 14,    0],
  lKnee:  [-18,   38],  rKnee:  [ 18,   42],
  lFoot:  [-22,   76],  rFoot:  [ 26,   80],
}

// ── JAB (lead hand = left) ────────────────────────────────────
// Chamber: lead hand draws back 8 units, shoulder rotates back
const JAB_CH = {
  head:   [ -1,  -98],  neck:   [ -1,  -80],
  lSh:    [-28,  -70],  rSh:    [ 22,  -68],
  lEl:    [-26,  -44],  rEl:    [ 16,  -44],
  lHand:  [-20,  -30],  rHand:  [  8,  -28],
  lHip:   [-16,    0],  rHip:   [ 12,    0],
  lKnee:  [-20,   38],  rKnee:  [ 16,   42],
  lFoot:  [-24,   76],  rFoot:  [ 24,   80],
}
// Attack: full extension, shoulder drives forward, body leans
const JAB_ATK = {
  head:   [  4,  -94],  neck:   [  3,  -77],
  lSh:    [-13,  -62],  rSh:    [ 26,  -71],
  lEl:    [ 18,  -61],  rEl:    [ 18,  -46],
  lHand:  [ 72,  -60],  rHand:  [ 10,  -30],
  lHip:   [-10,    1],  rHip:   [ 16,    1],
  lKnee:  [-14,   38],  rKnee:  [ 20,   42],
  lFoot:  [-18,   76],  rFoot:  [ 28,   80],
}
// Overshoot: 7 units past
const JAB_OVR = { ...JAB_ATK, lHand: [ 79, -60], lEl: [ 22, -61] }

// ── CROSS (rear hand = right) ─────────────────────────────────
// Chamber: rear shoulder loads, rear elbow goes back
const CROSS_CH = {
  head:   [  1,  -98],  neck:   [  1,  -80],
  lSh:    [-22,  -68],  rSh:    [ 28,  -70],
  lEl:    [-20,  -42],  rEl:    [ 24,  -50],
  lHand:  [-10,  -26],  rHand:  [ 18,  -36],
  lHip:   [-12,    0],  rHip:   [ 16,    0],
  lKnee:  [-16,   38],  rKnee:  [ 20,   42],
  lFoot:  [-20,   76],  rFoot:  [ 28,   80],
}
// Attack: rear shoulder DRIVES forward, lead pulls back, big hip rotation
const CROSS_ATK = {
  head:   [  5,  -93],  neck:   [  4,  -76],
  lSh:    [-30,  -73],  rSh:    [  8,  -59],
  lEl:    [-25,  -47],  rEl:    [ -2,  -58],
  lHand:  [-17,  -33],  rHand:  [ 70,  -57],
  lHip:   [ -7,    2],  rHip:   [ 19,    2],
  lKnee:  [-13,   36],  rKnee:  [ 23,   44],
  lFoot:  [-20,   76],  rFoot:  [ 30,   78],
}
const CROSS_OVR = { ...CROSS_ATK, rHand: [ 77, -57], rEl: [  3, -58] }

// ── HOOK (lead arm, horizontal arc) ──────────────────────────
// Chamber: elbow cocked WAY out to the side — elbow parallel to ground
const HOOK_CH = {
  head:   [  0,  -98],  neck:   [  0,  -80],
  lSh:    [-24,  -68],  rSh:    [ 22,  -68],
  lEl:    [-48,  -66],  rEl:    [ 16,  -44],
  lHand:  [-52,  -56],  rHand:  [  8,  -28],
  lHip:   [-16,    0],  rHip:   [ 12,    0],
  lKnee:  [-20,   38],  rKnee:  [ 14,   42],
  lFoot:  [-24,   76],  rFoot:  [ 22,   82],
}
// Attack: fist sweeps across at head height, elbow STAYS parallel to ground
// "Elbow at shoulder level" — the whole body rotates through it
const HOOK_ATK = {
  head:   [  5,  -94],  neck:   [  4,  -77],
  lSh:    [-14,  -62],  rSh:    [ 28,  -72],
  lEl:    [-12,  -62],  rEl:    [ 20,  -46],
  lHand:  [ 16,  -62],  rHand:  [ 12,  -30],
  lHip:   [ -7,    2],  rHip:   [ 19,    2],
  lKnee:  [-12,   36],  rKnee:  [ 22,   44],
  lFoot:  [-16,   76],  rFoot:  [ 28,   80],
}
const HOOK_OVR = { ...HOOK_ATK, lHand: [ 24, -62] }

// ── UPPERCUT (lead hand rises steeply) ───────────────────────
// Dip: deep knee bend, arm loads at hip level — "start low"
const UC_DIP = {
  head:   [  0,  -90],  neck:   [  0,  -72],
  lSh:    [-22,  -60],  rSh:    [ 20,  -60],
  lEl:    [-14,  -26],  rEl:    [ 14,  -38],
  lHand:  [ -2,  -10],  rHand:  [  8,  -22],
  lHip:   [-14,    8],  rHip:   [ 14,    8],
  lKnee:  [-18,   54],  rKnee:  [ 18,   58],
  lFoot:  [-22,   78],  rFoot:  [ 26,   82],
}
// Attack: explosive upward drive from legs — "drive through the legs"
const UC_ATK = {
  head:   [  2, -101],  neck:   [  2,  -83],
  lSh:    [-17,  -72],  rSh:    [ 24,  -70],
  lEl:    [ -7,  -56],  rEl:    [ 16,  -46],
  lHand:  [  6,  -98],  rHand:  [ 10,  -32],
  lHip:   [-10,   -2],  rHip:   [ 16,   -2],
  lKnee:  [-16,   36],  rKnee:  [ 18,   40],
  lFoot:  [-20,   76],  rFoot:  [ 26,   80],
}
const UC_OVR  = { ...UC_ATK, lHand: [  6, -106] }

// Idle breathing sway poses
const SWAY_L  = { ...GUARD, head: [-2,-98], lHip:[-16,1], rHip:[12,1], lKnee:[-20,39], rKnee:[16,43] }
const SWAY_R  = { ...GUARD, head: [ 2,-98], lHip:[-12,1], rHip:[16,1], lKnee:[-16,39], rKnee:[20,43] }

// ─────────────────────────────────────────────────────────────
// KEYFRAME TIMELINES
// Each entry: [normalizedTime, pose, easingToNext]
// Easing applies FROM this keyframe TO the next one.
//
// THE SNAP PATTERN (from tutorial):
//   [0.18, GUARD, 'io3']       ← hold in guard, then ease into chamber
//   [0.26, CHAMBER, 'in4']     ← easeInQuart: STAYS near chamber, SNAPS at end
//   [0.46, ATTACK, 'out2']     ← slight ease-out to overshoot
//   [0.52, OVERSHOOT, 'out3']  ← ease-out retract
//   [0.82, GUARD, 'io3']       ← settle back
// ─────────────────────────────────────────────────────────────
const KF = {
  jab: [
    [0.00, GUARD,     'io3'],
    [0.18, GUARD,     'io3'],   // hold in guard — anticipation window
    [0.27, JAB_CH,    'in4'],   // easeInQuart SNAP: frames cluster at chamber
    [0.47, JAB_ATK,   'out2'],  // arrive at attack, slight bounce
    [0.53, JAB_OVR,   'out3'],  // overshoot retracts
    [0.82, GUARD,     'io3'],
    [1.00, GUARD,     'linear'],
  ],
  cross: [
    [0.00, GUARD,      'io3'],
    [0.16, GUARD,      'io3'],
    [0.25, CROSS_CH,   'in4'],  // SNAP
    [0.47, CROSS_ATK,  'out2'],
    [0.54, CROSS_OVR,  'out3'],
    [0.84, GUARD,      'io3'],
    [1.00, GUARD,      'linear'],
  ],
  hook: [
    [0.00, GUARD,     'io3'],
    [0.14, GUARD,     'io3'],
    [0.26, HOOK_CH,   'in4'],   // SNAP — the cock-back makes anticipation very visible
    [0.50, HOOK_ATK,  'out2'],
    [0.57, HOOK_OVR,  'out3'],
    [0.86, GUARD,     'io3'],
    [1.00, GUARD,     'linear'],
  ],
  uppercut: [
    [0.00, GUARD,    'io3'],
    [0.22, UC_DIP,   'in4'],    // SNAP — dip then explode
    [0.48, UC_ATK,   'out2'],
    [0.55, UC_OVR,   'out3'],
    [0.84, GUARD,    'io3'],
    [1.00, GUARD,    'linear'],
  ],
  idle: [
    [0.00, GUARD,   'io2'],
    [0.25, SWAY_L,  'io2'],
    [0.50, GUARD,   'io2'],
    [0.75, SWAY_R,  'io2'],
    [1.00, GUARD,   'linear'],
  ],
}

// ─────────────────────────────────────────────────────────────
// POSE INTERPOLATION
// ─────────────────────────────────────────────────────────────
function lerp(a, b, t) {
  return a + (b - a) * t
}

function lerpPose(p0, p1, t) {
  const out = {}
  for (const k of Object.keys(p0)) {
    out[k] = [lerp(p0[k][0], p1[k][0], t), lerp(p0[k][1], p1[k][1], t)]
  }
  return out
}

function samplePose(moveName, t) {
  const kfs = KF[moveName] || KF.idle
  let i = 0
  while (i < kfs.length - 2 && t >= kfs[i + 1][0]) i++
  const [t0, pose0, easing] = kfs[i]
  const [t1, pose1]         = kfs[i + 1]
  if (t1 <= t0) return pose0
  const lt = Math.max(0, Math.min(1, (t - t0) / (t1 - t0)))
  return lerpPose(pose0, pose1, (E[easing] || E.linear)(lt))
}

// ─────────────────────────────────────────────────────────────
// DRAWING
// ─────────────────────────────────────────────────────────────

// Convert joint offset to canvas position
function pt(pose, k, cx, cy, s) {
  return [cx + pose[k][0] * s, cy + pose[k][1] * s]
}

// Thick limb line with subtle glow (two passes — glow then core)
function limb(ctx, p1, p2, col, w) {
  ctx.lineCap = 'round'
  ctx.strokeStyle = col + '22'
  ctx.lineWidth   = w + 8
  ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke()
  ctx.strokeStyle = col
  ctx.lineWidth   = w
  ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke()
}

// Filled joint circle
function dot(ctx, pos, r, col) {
  ctx.fillStyle = col
  ctx.beginPath(); ctx.arc(...pos, r, 0, Math.PI * 2); ctx.fill()
}

// Glove: colored filled circle + impact ring
function glove(ctx, pos, r, col, flash) {
  if (flash > 0.02) {
    // Outer expanding ring
    ctx.save()
    ctx.globalAlpha = flash * 0.9
    ctx.strokeStyle = col; ctx.lineWidth = 2.5
    ctx.beginPath(); ctx.arc(...pos, r + 5 + flash * 16, 0, Math.PI * 2); ctx.stroke()
    // White inner ring
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.globalAlpha = flash * 0.55
    ctx.beginPath(); ctx.arc(...pos, r + 2 + flash * 7, 0, Math.PI * 2); ctx.stroke()
    ctx.restore()
  }
  // Soft glow fill
  ctx.save()
  ctx.globalAlpha = 0.28
  ctx.fillStyle   = col
  ctx.beginPath(); ctx.arc(...pos, r + 6, 0, Math.PI * 2); ctx.fill()
  ctx.restore()
  // Core glove
  ctx.fillStyle = col
  ctx.beginPath(); ctx.arc(...pos, r, 0, Math.PI * 2); ctx.fill()
}

// Speed lines behind the fist during snap — 4 short radial lines
function speedLines(ctx, pos, dir, col, strength) {
  if (strength < 0.15) return
  ctx.save()
  ctx.globalAlpha = strength * 0.65
  ctx.strokeStyle = col; ctx.lineCap = 'round'; ctx.lineWidth = 1.5
  const [dx, dy] = dir
  const len = 22 * strength
  // 4 lines fanning out behind the fist
  for (const spread of [-0.35, -0.12, 0.12, 0.35]) {
    const angle  = Math.atan2(dy, dx) + spread
    const ox     = Math.cos(angle) * len, oy = Math.sin(angle) * len
    const startX = pos[0] - ox * 1.8,   startY = pos[1] - oy * 1.8
    ctx.beginPath()
    ctx.moveTo(startX, startY)
    ctx.lineTo(startX - ox * 0.8, startY - oy * 0.8)
    ctx.stroke()
  }
  ctx.restore()
}

// ─────────────────────────────────────────────────────────────
// FULL FIGURE DRAW
// Per move:
//   jab, hook, uppercut → left arm is active (lead, drawn on top)
//   cross               → right arm is active (rear but draws on top)
// ─────────────────────────────────────────────────────────────
const FRONT   = '#d8e4ff'   // front/lead limbs — bright
const REAR    = '#28384e'   // rear limbs — dim (depth)
const SPINE_C = '#6a7a9a'   // torso / spine
const JDIM    = '#1e2c40'   // very dim rear joints

function drawFigure(ctx, pose, cx, cy, s, col, flash, activeHand, snapDir) {
  const p = k => pt(pose, k, cx, cy, s)

  // ── Floor shadow ─────────────────────────────────────────
  ctx.save(); ctx.globalAlpha = 0.08; ctx.fillStyle = SPINE_C
  const fx = (p('lFoot')[0] + p('rFoot')[0]) * 0.5
  const fy = Math.max(p('lFoot')[1], p('rFoot')[1]) + 7 * s
  ctx.beginPath(); ctx.ellipse(fx, fy, 28 * s, 5 * s, 0, 0, Math.PI * 2); ctx.fill()
  ctx.restore()

  // ── Rear leg (right) ─────────────────────────────────────
  limb(ctx, p('rHip'), p('rKnee'), REAR, 3.5)
  limb(ctx, p('rKnee'), p('rFoot'), REAR, 3.5)
  // Rear foot line
  ctx.strokeStyle = REAR; ctx.lineWidth = 2; ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(p('rFoot')[0] - 4 * s, p('rFoot')[1])
  ctx.lineTo(p('rFoot')[0] + 14 * s, p('rFoot')[1])
  ctx.stroke()
  dot(ctx, p('rKnee'), 4, REAR)

  // ── Torso fill (low opacity — depth/volume) ────────────── 
  ctx.save(); ctx.globalAlpha = 0.07; ctx.fillStyle = SPINE_C
  ctx.beginPath()
  ctx.moveTo(...p('lHip')); ctx.lineTo(...p('rHip'))
  ctx.lineTo(...p('rSh'));  ctx.lineTo(...p('lSh'))
  ctx.closePath(); ctx.fill()
  ctx.restore()

  // ── Spine & shoulder line ─────────────────────────────────
  const hipMid = [
    (p('lHip')[0] + p('rHip')[0]) * 0.5,
    (p('lHip')[1] + p('rHip')[1]) * 0.5,
  ]
  ctx.strokeStyle = SPINE_C; ctx.lineWidth = 2; ctx.lineCap = 'round'
  ctx.beginPath(); ctx.moveTo(...p('lHip')); ctx.lineTo(...p('rHip')); ctx.stroke()
  limb(ctx, hipMid, p('neck'), SPINE_C, 3)
  limb(ctx, p('lSh'), p('rSh'), SPINE_C, 2.5)

  // ── Front leg (left / lead) ──────────────────────────────
  limb(ctx, p('lHip'),  p('lKnee'), FRONT, 4.5)
  limb(ctx, p('lKnee'), p('lFoot'), FRONT, 4.5)
  ctx.strokeStyle = FRONT; ctx.lineWidth = 2.5; ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(p('lFoot')[0] - 8 * s, p('lFoot')[1])
  ctx.lineTo(p('lFoot')[0] + 14 * s, p('lFoot')[1])
  ctx.stroke()
  dot(ctx, p('lKnee'), 4.5, SPINE_C)

  // ── Head ─────────────────────────────────────────────────
  limb(ctx, p('neck'), p('head'), FRONT, 2.5)
  ctx.strokeStyle = FRONT; ctx.lineWidth = 2.5
  ctx.beginPath(); ctx.arc(...p('head'), 12, 0, Math.PI * 2); ctx.stroke()

  // ── Determine arm draw order (active arm drawn on top) ───
  const isRearActive = activeHand === 'rHand'
  const firstArm  = isRearActive ? 'l' : 'r'  // non-active drawn first
  const secondArm = isRearActive ? 'r' : 'l'  // active drawn on top

  const drawArm = (side, isActive) => {
    const sh = p(side + 'Sh'), el = p(side + 'El'), hand = p(side + 'Hand')
    const c  = isActive ? col : (side === 'r' ? REAR : SPINE_C)
    const w  = isActive ? 4.5 : 3

    limb(ctx, sh,  el,   c, w)
    limb(ctx, el,  hand, c, w)
    dot(ctx, el, isActive ? 5.5 : 4, c)

    if (isActive) {
      // Speed lines during snap (before impact)
      if (flash < 0.3 && snapDir) {
        speedLines(ctx, hand, snapDir, col, (1 - flash / 0.3) * 0.6)
      }
      glove(ctx, hand, 9, col, flash)
    } else {
      // Non-active: dimmer glove circle
      ctx.fillStyle = c
      ctx.beginPath(); ctx.arc(...hand, 7, 0, Math.PI * 2); ctx.fill()
    }
  }

  drawArm(firstArm,  false)
  drawArm(secondArm, true)

  // ── Shoulder/hip joint dots ──────────────────────────────
  for (const k of ['lSh', 'rSh', 'lHip', 'rHip']) {
    dot(ctx, p(k), k.startsWith('l') ? 5.5 : 4, k.startsWith('l') ? SPINE_C : JDIM)
  }
}

// ─────────────────────────────────────────────────────────────
// TRAIL (circular buffer, N positions of the active glove)
// ─────────────────────────────────────────────────────────────
const TRAIL_CAP = 14

function drawTrail(ctx, trail, col) {
  if (trail.length < 2) return
  trail.forEach((pos, i) => {
    const frac = i / (trail.length - 1)
    ctx.save()
    ctx.globalAlpha  = frac * 0.4
    ctx.fillStyle    = col
    ctx.beginPath(); ctx.arc(...pos, 3 + frac * 7, 0, Math.PI * 2); ctx.fill()
    ctx.restore()
  })
}

// ─────────────────────────────────────────────────────────────
// BACKGROUND (grid baked offscreen, gradient cached)
// ─────────────────────────────────────────────────────────────
let _gridCache = null, _gW = 0, _gH = 0
function getGrid(W, H) {
  if (_gridCache && _gW === W && _gH === H) return _gridCache
  const oc = document.createElement('canvas'); oc.width = W; oc.height = H
  const c  = oc.getContext('2d')
  c.strokeStyle = 'rgba(0,220,255,0.028)'; c.lineWidth = 1
  for (let x = 0; x < W; x += 28) { c.beginPath(); c.moveTo(x,0); c.lineTo(x,H); c.stroke() }
  for (let y = 0; y < H; y += 28) { c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke() }
  _gridCache = oc; _gW = W; _gH = H; return oc
}

const _bgCache = {}
function getBg(ctx, cx, cy, W, H, col) {
  const key = `${Math.round(cx)},${W},${H},${col}`
  if (!_bgCache[key]) {
    const g = ctx.createRadialGradient(cx, cy*0.5, 0, cx, cy*0.5, Math.max(W,H)*0.8)
    g.addColorStop(0,   col + '10')
    g.addColorStop(0.5, col + '04')
    g.addColorStop(1,   'transparent')
    _bgCache[key] = g
  }
  return _bgCache[key]
}

// ─────────────────────────────────────────────────────────────
// MOVE METADATA
// ─────────────────────────────────────────────────────────────
const META = {
  jab:      { color:'#ffea00', name:'JAB',      phases:['CHAMBER','SNAP','RETRACT'], tip:'Lead hand. Chamber → snap out → snap back instantly.',         activeHand:'lHand', impactT:0.47 },
  cross:    { color:'#00dcff', name:'CROSS',     phases:['PIVOT',  'DRIVE','RESET'],  tip:'Pivot rear foot. Hips → shoulder → fist. Full rotation.',      activeHand:'rHand', impactT:0.47 },
  hook:     { color:'#00e676', name:'HOOK',      phases:['COCK',   'SWEEP','RETURN'], tip:'Elbow parallel to ground. Sweep the whole body through.',       activeHand:'lHand', impactT:0.50 },
  uppercut: { color:'#d500f9', name:'UPPERCUT',  phases:['DIP',    'EXPLODE','GUARD'],tip:'Dip deep. Drive up from the legs. Straight explosive rise.',    activeHand:'lHand', impactT:0.48 },
  idle:     { color:'#4466aa', name:'GUARD',     phases:['PROTECT','BREATHE','READY'],tip:'Orthodox high guard. Weight on balls of feet. Eyes on target.', activeHand:'lHand', impactT:-1   },
  demo:     { color:'#ffea00', name:'JAB',       phases:['CHAMBER','SNAP','RETRACT'], tip:'Lead hand. Chamber → snap out → snap back instantly.',         activeHand:'lHand', impactT:0.47 },
}

const DEMO_SEQ  = ['jab', 'cross', 'hook', 'uppercut']
const MOVE_SECS = 2.0

// ─────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────
export default function TechniqueAnimator({ move = 'idle' }) {
  const canvasRef    = useRef(null)
  const rafRef       = useRef(null)
  const startRef     = useRef(performance.now())
  const trailRef     = useRef([])
  const prevHandRef  = useRef(null)
  const activeRef    = useRef(move === 'demo' ? DEMO_SEQ[0] : move)
  const demoIdx      = useRef(0)

  const [display, setDisplay] = useState(activeRef.current)
  const [phase,   setPhase]   = useState(0)

  useEffect(() => {
    startRef.current = performance.now()
    trailRef.current = []
    prevHandRef.current = null
    demoIdx.current   = 0
    activeRef.current = move === 'demo' ? DEMO_SEQ[0] : move
    setDisplay(activeRef.current)

    const loop = (now) => {
      const elapsed = (now - startRef.current) / 1000
      const t       = Math.min(elapsed / MOVE_SECS, 0.9999)

      if (elapsed >= MOVE_SECS) {
        trailRef.current    = []
        prevHandRef.current = null
        startRef.current    = now
        if (move === 'demo') {
          const next        = (demoIdx.current + 1) % DEMO_SEQ.length
          demoIdx.current   = next
          activeRef.current = DEMO_SEQ[next]
          setDisplay(DEMO_SEQ[next])
        }
        rafRef.current = requestAnimationFrame(loop)
        return
      }

      const ph = t < 0.33 ? 0 : t < 0.66 ? 1 : 2
      setPhase(prev => prev !== ph ? ph : prev)

      const cvs = canvasRef.current
      if (!cvs) { rafRef.current = requestAnimationFrame(loop); return }
      const ctx = cvs.getContext('2d')
      const W   = cvs.width  = cvs.offsetWidth
      const H   = cvs.height = cvs.offsetHeight
      if (W < 1 || H < 1) { rafRef.current = requestAnimationFrame(loop); return }

      const activeName = activeRef.current
      const meta       = META[activeName] || META.idle
      const col        = meta.color
      const cx         = W * 0.5
      const cy         = H * 0.56
      const s          = Math.min(W, H) / 400

      // Background
      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = getBg(ctx, cx, cy, W, H, col)
      ctx.fillRect(0, 0, W, H)
      ctx.drawImage(getGrid(W, H), 0, 0)

      // Sample pose
      const pose = samplePose(activeName, t)

      // Update trail
      const handPos = pt(pose, meta.activeHand, cx, cy, s)
      // Only add to trail during the snap/attack window (not during guard hold)
      if (t > 0.28 && t < 0.70) {
        trailRef.current.push([...handPos])
        if (trailRef.current.length > TRAIL_CAP) trailRef.current.shift()
      }

      // Compute snap direction for speed lines (vector from prev hand to current)
      let snapDir = null
      if (prevHandRef.current) {
        const [px, py] = prevHandRef.current
        const dx = handPos[0] - px, dy = handPos[1] - py
        const d  = Math.sqrt(dx*dx + dy*dy)
        if (d > 0.5) snapDir = [dx / d, dy / d]
      }
      prevHandRef.current = [...handPos]

      // Flash at impact moment
      const flash = Math.max(0, 1 - Math.abs(t - meta.impactT) / 0.055)

      // Draw trail under figure
      drawTrail(ctx, trailRef.current, col)

      // Draw figure
      ctx.save()
      ctx.lineCap  = 'round'
      ctx.lineJoin = 'round'
      drawFigure(ctx, pose, cx, cy, s, col, flash, meta.activeHand, snapDir)
      ctx.restore()

      rafRef.current = requestAnimationFrame(loop)
    }

    rafRef.current = requestAnimationFrame(loop)
    return () => { cancelAnimationFrame(rafRef.current); rafRef.current = null }
  }, [move])

  const meta = META[display] || META.idle

  return (
    <div className={styles.wrap}>
      <canvas ref={canvasRef} className={styles.canvas} />
      <div className={styles.info}>
        <div className={styles.moveName}
          style={{ color: meta.color, textShadow: `0 0 22px ${meta.color}88` }}>
          {meta.name}
        </div>
        <div className={styles.tip}>{meta.tip}</div>
        <div className={styles.phases}>
          {meta.phases.map((label, i) => (
            <div key={i}
              className={`${styles.phaseTag} ${phase === i ? styles.phaseActive : ''}`}
              style={phase === i ? { color: meta.color, borderColor: meta.color } : {}}>
              {label}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
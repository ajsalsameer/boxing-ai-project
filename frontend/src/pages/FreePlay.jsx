import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useWS } from '../context/WSContext'
import CameraFeed from '../components/CameraFeed'
import TechniqueAnimator from '../components/TechniqueAnimator'
import ErrorBoundary from '../components/ErrorBoundary'
import styles from './FreePlay.module.css'

// Punch classes only — used for the main 4 bars
const PUNCHES = ['jab', 'cross', 'hook', 'uppercut']
// All 5 classes — matches server probs array order
const ALL_CLASSES = ['jab', 'cross', 'hook', 'uppercut', 'idle']

const COLORS = {
  jab: '#ffea00',
  cross: '#00dcff',
  hook: '#00e676',
  uppercut: '#d500f9',
  idle: '#334466',   // was '#334' — 3-digit hex is invalid for canvas API
}
const LABELS = {
  jab: 'JAB', cross: 'CROSS', hook: 'HOOK', uppercut: 'UPPERCUT', idle: '—',
}

const CONFIRM_THRESHOLD = 0.68   // matches server (was 0.72, lowered because idle absorbs FPs)
const HOLD_MS = 1800

// ── Confidence bar ────────────────────────────────────────────
function ConfBar({ name, color, value, isTop, isIdle }) {
  return (
    <div className={`${styles.barRow} ${isIdle ? styles.barRowIdle : ''}`}>
      <div className={styles.barName}
        style={{ color: isTop ? color : isIdle ? '#2a3450' : '#444' }}>
        {name}
      </div>
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{
            width: `${Math.round(value * 100)}%`,
            background: isTop ? color : isIdle ? 'rgba(51,68,102,0.35)' : 'rgba(255,255,255,0.06)',
            transition: 'none',
          }}
        />
      </div>
      <div className={styles.barVal}
        style={{ color: isTop ? color : isIdle ? '#2a3450' : '#333' }}>
        {Math.round(value * 100)}%
      </div>
    </div>
  )
}

// ── Markov quality badge ──────────────────────────────────────
// Shows how much data the Markov chain has built up.
// Green = reliable predictions, grey = still learning.
function MarkovBadge({ stats }) {
  if (!stats) return null
  const n = stats.total_transitions || 0
  const col = n >= 30 ? '#00e676' : n >= 10 ? '#00dcff' : '#334466'
  const lbl = n >= 30 ? 'MARKOV ✓' : n >= 10 ? `MARKOV ~${n}` : 'MARKOV LEARNING'
  return (
    <span className={styles.markovBadge} style={{ color: col, borderColor: col + '44' }}>
      {lbl}
    </span>
  )
}

// ── Main page ─────────────────────────────────────────────────
export default function FreePlay() {
  const { inference, cameraActive, startCamera, stopCamera, currentUser, connected } = useWS()
  const nav = useNavigate()

  // probs is now 5 elements: [jab, cross, hook, uppercut, idle]
  const [confirmed, setConfirmed] = useState({
    move: 'idle', probs: [0, 0, 0, 0, 0], topConf: 0, flashKey: 0,
  })
  const [showing, setShowing] = useState(false)
  const [animMove, setAnimMove] = useState('idle')
  const [confirmedNext, setConfirmedNext] = useState({ move: null, conf: 0 })
  const [markovStats, setMarkovStats] = useState(null)
  const [animEnabled, setAnimEnabled] = useState(false)

  const holdTimer = useRef(null)
  const prevMove = useRef('idle')

  useEffect(() => {
    if (!inference) return

    const move = inference.move || 'idle'
    const probs = inference.probs || [0, 0, 0, 0, 0]   // 5 elements
    // topConf from all 5 classes, but we only confirm punches
    const punchProbs = probs.slice(0, 4)
    const topConf = punchProbs.length ? Math.max(...punchProbs) : 0

    // Update Markov stats whenever we get a new inference
    if (inference.markov_stats) {
      setMarkovStats(inference.markov_stats)
    }

    // Only confirm a punch (not idle) above threshold
    if (move !== 'idle' && move !== prevMove.current && topConf >= CONFIRM_THRESHOLD) {
      prevMove.current = move
      setConfirmed(prev => ({ move, probs, topConf, flashKey: prev.flashKey + 1 }))
      setShowing(true)
      setAnimMove(move)

      if (inference.next_move && inference.next_move !== 'idle') {
        setConfirmedNext({ move: inference.next_move, conf: inference.next_conf || 0 })
      } else {
        setConfirmedNext({ move: null, conf: 0 })
      }

      if (holdTimer.current) clearTimeout(holdTimer.current)
      holdTimer.current = setTimeout(() => {
        setShowing(false)
        prevMove.current = 'idle'
      }, HOLD_MS)
    }

    if (move === 'idle' && prevMove.current !== 'idle') {
      prevMove.current = 'idle'
    }
  }, [inference])

  useEffect(() => () => { if (holdTimer.current) clearTimeout(holdTimer.current) }, [])

  const displayMove = showing ? confirmed.move : 'idle'
  const displayProbs = confirmed.probs   // 5 elements
  const displayConf = confirmed.topConf
  const color = COLORS[displayMove] || COLORS.idle
  const combo = inference?.combo || []
  const fps = inference?.fps || 0
  const markovLabel = inference?.markov_label || null

  // topIdx among PUNCH classes only (indices 0-3), never highlight idle as top
  const punchProbs = displayProbs.slice(0, 4)
  const topIdx = confirmed.move !== 'idle'
    ? punchProbs.indexOf(Math.max(...punchProbs))
    : -1

  const idleProb = displayProbs[4] || 0

  return (
    <div className={styles.wrapper}>

      {!connected && (
        <div className={styles.offlineBanner}>
          ⚠ Backend offline — run <code>python server.py</code> and this page will connect automatically
        </div>
      )}

      <div className={styles.page} style={!connected ? { opacity: 0.35, pointerEvents: 'none' } : {}}>

        {/* ── LEFT: Camera ─────────────────────────────── */}
        <div className={styles.camCol}>
          <div className={styles.colHeader}>
            <span className={styles.colLabel}>▶ LIVE FEED</span>
            <span className={styles.fps}>{fps} FPS</span>
          </div>
          <div className={styles.camWrap}><CameraFeed /></div>
          <div className={styles.camFooter}>
            <button
              className={`hex-btn ${cameraActive ? 'hex-btn-red' : 'hex-btn-green'}`}
              onClick={cameraActive ? stopCamera : startCamera}
            >
              {cameraActive ? '⬛ STOP' : '▶ ACTIVATE CAMERA'}
            </button>
          </div>
        </div>

        {/* ── CENTER: HUD ──────────────────────────────── */}
        <div className={styles.hudCol}>

          {!currentUser ? (
            <div className={styles.noUserBanner} onClick={() => nav('/training')}>
              ⚠ No profile — predictions use global model.
              <span className={styles.noUserLink}> Set up profile →</span>
            </div>
          ) : (
            <div className={styles.userBanner}>
              <span>👤 {currentUser.username}</span>
              <div className={styles.bannerRight}>
                <MarkovBadge stats={markovStats} />
                <span className={styles.gruLabel}>
                  {currentUser.summary?.has_model ? '🧠 Personal model' : '⚠ Global model'}
                </span>
              </div>
            </div>
          )}

          {/* Confirmed move label */}
          <div
            className={`glass-panel ${styles.moveBox} ${showing ? styles.moveActive : ''}`}
            key={confirmed.flashKey}
          >
            <div className={styles.moveLabel}>DETECTED MOVE</div>
            <div
              className={styles.moveText}
              style={{
                color,
                textShadow: showing ? `0 0 30px ${color}, 0 0 60px ${color}66` : 'none',
              }}
            >
              {showing ? (LABELS[displayMove] || '—') : '—'}
            </div>
            {showing && (
              <div className={styles.confBadge} style={{ color, borderColor: `${color}55` }}>
                {Math.round(displayConf * 100)}%
              </div>
            )}
            <div
              className={styles.moveLine}
              style={{
                background: showing ? color : 'rgba(255,255,255,0.05)',
                boxShadow: showing ? `0 0 12px ${color}` : 'none',
              }}
            />
          </div>

          {/* Frozen prob snapshot — 4 punch bars + 1 idle bar */}
          <div className={`glass-panel ${styles.barsBox}`}>
            <div className="hud-label" style={{ padding: '12px 16px 6px' }}>
              LAST CONFIRMED
              {confirmed.move !== 'idle' && (
                <span style={{ color: COLORS[confirmed.move], marginLeft: 8, fontSize: 9 }}>
                  {confirmed.move.toUpperCase()}
                </span>
              )}
            </div>
            <div className={styles.bars}>
              {/* 4 punch bars */}
              {PUNCHES.map((m, i) => (
                <ConfBar
                  key={m}
                  name={m.toUpperCase()}
                  color={COLORS[m]}
                  value={displayProbs[i] || 0}
                  isTop={i === topIdx && confirmed.move !== 'idle'}
                  isIdle={false}
                />
              ))}
              {/* Idle bar — dimmer, separated visually */}
              <ConfBar
                key="idle"
                name="IDLE"
                color={COLORS.idle}
                value={idleProb}
                isTop={false}
                isIdle={true}
              />
            </div>
            {confirmed.move === 'idle' && (
              <div className={styles.waitingMsg}>
                {idleProb > 0.6
                  ? 'model sees idle — stand still ✓'
                  : 'waiting for first punch...'}
              </div>
            )}
          </div>

          {/* Next prediction (Markov) */}
          <div className={`glass-panel ${styles.predBox}`}>
            <div className="hud-label" style={{ padding: '12px 16px 6px' }}>◈ NEXT PREDICTION</div>
            {confirmedNext.move && confirmedNext.move !== 'idle'
              ? <>
                <div
                  className={styles.predMove}
                  style={{
                    color: COLORS[confirmedNext.move],
                    textShadow: `0 0 16px ${COLORS[confirmedNext.move]}`,
                  }}
                >
                  {confirmedNext.move.toUpperCase()}
                </div>
                <div className={styles.predConf}>
                  {Math.round(confirmedNext.conf * 100)}% confidence
                </div>
                {markovLabel && (
                  <div className={styles.predModel}>
                    via {markovLabel} markov
                  </div>
                )}
              </>
              : <div className={styles.predEmpty}>
                {markovStats && markovStats.total_transitions < 10
                  ? `${markovStats.total_transitions}/10 punches to activate`
                  : '—'}
              </div>
            }
          </div>

          {/* Combo strip */}
          <div className={`glass-panel ${styles.comboBox}`}>
            <div className="hud-label" style={{ padding: '12px 16px 8px' }}>◈ COMBO</div>
            <div className={styles.comboStrip}>
              {combo.length === 0
                ? <span className={styles.comboEmpty}>throw a punch to start</span>
                : combo.map((m, i) => (
                  <span key={i} className={styles.comboPill}
                    style={{ color: COLORS[m], borderColor: COLORS[m], background: `${COLORS[m]}12` }}>
                    {m.toUpperCase()}
                  </span>
                ))
              }
            </div>
          </div>
        </div>

        {/* ── RIGHT: Technique ─────────────────────────── */}
        <div className={styles.techCol}>
          <div className={styles.colHeader}>
            <span className={styles.colLabel}>◈ TECHNIQUE REFERENCE</span>
            <button
              className={styles.animToggle}
              style={{
                color: animEnabled ? '#00dcff' : '#334466',
                borderColor: animEnabled ? '#00dcff44' : '#33446644',
                background: animEnabled ? 'rgba(0,220,255,0.06)' : 'transparent',
              }}
              onClick={() => setAnimEnabled(v => !v)}
            >
              {animEnabled ? '◈ ON' : '◈ OFF'}
            </button>
          </div>
          <div className={styles.techWrap}>
            {animEnabled
              ? <ErrorBoundary label="Technique Animator">
                <TechniqueAnimator move={animMove} />
              </ErrorBoundary>
              : <div className={styles.animOff}>
                <div className={styles.animOffIcon}>◈</div>
                <div className={styles.animOffText}>ANIMATION OFF</div>
                <div className={styles.animOffSub}>toggle to enable</div>
              </div>
            }
          </div>
        </div>

      </div>
    </div>
  )
}
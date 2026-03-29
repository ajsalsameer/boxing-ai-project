import { useEffect, useState, useRef, useCallback } from 'react'
import { useWS } from '../context/WSContext'
import CameraFeed from '../components/CameraFeed'
import TechniqueAnimator from '../components/TechniqueAnimator'
import ErrorBoundary from '../components/ErrorBoundary'
import styles from './Coach.module.css'

// Punch colours only — idle is never a coach command
const COLORS = {
  jab: '#ffea00',
  cross: '#00dcff',
  hook: '#00e676',
  uppercut: '#d500f9',
}

function StatBox({ label, value, color }) {
  return (
    <div className={styles.statBox}>
      <div className={styles.statLabel}>{label}</div>
      <div className={styles.statVal}
        style={color ? { color, textShadow: `0 0 12px ${color}` } : {}}>
        {value}
      </div>
    </div>
  )
}

export default function Coach() {
  const { inference, cameraActive, startCamera, stopCamera, send, connected } = useWS()
  const [coachActive, setCoachActive] = useState(false)
  const [feedbackVisible, setFeedbackVisible] = useState(false)
  const [animEnabled, setAnimEnabled] = useState(false)
  const feedbackTimer = useRef(null)

  const coach = inference?.coach || {}
  const command = coach.command              // always a punch name or null — never 'idle'
  const animMove = coach.anim || 'idle'
  const feedback = coach.feedback || ''
  const fbColor = coach.fb_color || '#4caf50'
  const stats = coach.stats || {}
  const phase = coach.phase
  const elapsed = coach.cmd_elapsed || 0

  // Guard: command should never be 'idle' from the server, but be safe
  const safeCommand = command && command !== 'idle' ? command : null
  const cmdColor = safeCommand ? (COLORS[safeCommand] || '#fff') : '#333'

  useEffect(() => {
    if (feedback) {
      setFeedbackVisible(true)
      clearTimeout(feedbackTimer.current)
      feedbackTimer.current = setTimeout(() => setFeedbackVisible(false), 4200)
    }
  }, [feedback])

  const handleStart = useCallback(async () => {
    if (!cameraActive) {
      const ok = await startCamera()
      if (!ok) return
    }
    setCoachActive(true)
    send({ type: 'set_mode', mode: 'coach' })
  }, [cameraActive, startCamera, send])

  const handleStop = useCallback(() => {
    setCoachActive(false)
    send({ type: 'coach_stop' })
  }, [send])

  const isDone = phase === 'done'

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
          <div className={styles.colLabel}>▶ LIVE FEED</div>
          <div className={styles.camWrap}><CameraFeed /></div>
          <div className={styles.camFooter}>
            {!coachActive ? (
              <button className="hex-btn hex-btn-green"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={handleStart}>
                ⚔ BEGIN TRAINING SESSION
              </button>
            ) : (
              <button className="hex-btn hex-btn-red"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={handleStop}>
                ⬛ END SESSION
              </button>
            )}
            {cameraActive && coachActive && (
              <button className={`hex-btn hex-btn-blue ${styles.stopCam}`} onClick={stopCamera}>
                STOP CAMERA
              </button>
            )}
          </div>
        </div>

        {/* ── CENTER: Coach HUD ────────────────────────── */}
        <div className={styles.hudCol}>

          <div className={`glass-panel ${styles.cmdBox}`}
            style={{
              borderColor: coachActive && safeCommand ? cmdColor + '55' : 'var(--c-border)',
              boxShadow: coachActive && safeCommand ? `0 0 30px ${cmdColor}22` : 'none',
            }}>
            <div className={styles.cmdLabel}>
              {coachActive ? '◉ COMMAND' : 'READY TO TRAIN?'}
            </div>
            {!coachActive ? (
              <div className={styles.cmdIdle}>—</div>
            ) : isDone ? (
              <div className={styles.cmdDone}>COMPLETE!</div>
            ) : safeCommand ? (
              <>
                <div className={styles.cmdText}
                  style={{
                    color: cmdColor,
                    textShadow: `0 0 30px ${cmdColor}, 0 0 60px ${cmdColor}55`,
                  }}>
                  {safeCommand.toUpperCase()}
                </div>
                <div className={styles.cmdTimer}>SHOWING FOR {elapsed}s</div>
              </>
            ) : (
              <div className={styles.cmdWait}>LOADING…</div>
            )}
          </div>

          <div className={`glass-panel ${styles.feedbackBox} ${feedbackVisible ? styles.feedbackVisible : ''}`}>
            {feedbackVisible && (
              <div className={styles.feedbackText}
                style={{ color: fbColor, borderColor: fbColor, background: `${fbColor}10` }}>
                {feedback}
              </div>
            )}
          </div>

          <div className={styles.statsGrid}>
            <StatBox label="REPS" value={stats.correct ?? '—'}
              color={stats.correct > 0 ? 'var(--neon-green)' : null} />
            <StatBox label="ACCURACY" value={stats.accuracy > 0 ? `${stats.accuracy}%` : '—'}
              color={stats.accuracy >= 80 ? 'var(--neon-green)' : stats.accuracy >= 60 ? 'var(--neon-yellow)' : null} />
            <StatBox label="AVG TIME" value={stats.avg_time > 0 ? `${stats.avg_time}s` : '—'} />
            <StatBox label="MISTAKES" value={stats.mistakes ?? '—'}
              color={stats.mistakes > 0 ? 'var(--neon-red)' : null} />
          </div>

          {!coachActive && (
            <div className={`glass-panel ${styles.hintBox}`}>
              <div className={styles.hintTitle}>HOW IT WORKS</div>
              <div className={styles.hints}>
                <div className={styles.hintRow}><span className={styles.hintNum}>01</span>Press BEGIN TRAINING</div>
                <div className={styles.hintRow}><span className={styles.hintNum}>02</span>A punch command appears</div>
                <div className={styles.hintRow}><span className={styles.hintNum}>03</span>Throw the correct punch</div>
                <div className={styles.hintRow}><span className={styles.hintNum}>04</span>AI grades speed & accuracy</div>
                <div className={styles.hintRow}><span className={styles.hintNum}>05</span>Complete all combos to finish</div>
              </div>
            </div>
          )}

          {isDone && (
            <div className={`glass-panel ${styles.doneBox}`}>
              <div className={styles.doneTitle}>🏆 SESSION COMPLETE</div>
              <div className={styles.doneStats}>
                <div>Correct reps: <strong style={{ color: 'var(--neon-green)' }}>{stats.correct}</strong></div>
                <div>Accuracy: <strong style={{ color: 'var(--neon-yellow)' }}>{stats.accuracy}%</strong></div>
                <div>Avg reaction: <strong style={{ color: 'var(--neon-blue)' }}>{stats.avg_time}s</strong></div>
              </div>
              <button className="hex-btn hex-btn-green"
                style={{ marginTop: 14, width: '100%', justifyContent: 'center' }}
                onClick={handleStart}>
                ⟳ RESTART SESSION
              </button>
            </div>
          )}

        </div>

        {/* ── RIGHT: Technique ─────────────────────────── */}
        <div className={styles.techCol}>
          <div className={styles.techColHeader}>
            <div className={styles.colLabel}>◈ TECHNIQUE REFERENCE</div>
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
                <TechniqueAnimator move={coachActive ? animMove : 'idle'} />
              </ErrorBoundary>
              : <div className={styles.animOff}>
                <div className={styles.animOffIcon}>◈</div>
                <div className={styles.animOffText}>ANIMATION OFF</div>
                <div className={styles.animOffSub}>toggle to enable</div>
              </div>
            }
          </div>
          {animEnabled && coachActive && safeCommand && (
            <div className={styles.techHint}>
              Watch the animation · then throw your {safeCommand.toUpperCase()}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
import { useState, useEffect, useRef } from 'react'
import { useWS } from '../context/WSContext'
import CameraFeed from '../components/CameraFeed'
import styles from './Training.module.css'

const STEPS = ['LOGIN', 'RECORD', 'TRAIN', 'READY']

export default function Training() {
  const {
    connected, inference, cameraActive, startCamera, stopCamera,
    currentUser, loginUser, startRecording, stopRecording, trainPersonal,
    trainProgress, wsMessages,
  } = useWS()

  const [step,         setStep]         = useState('LOGIN')   // LOGIN | RECORD | TRAIN | READY
  const [username,     setUsername]     = useState('')
  const [inputErr,     setInputErr]     = useState('')
  const [recActive,    setRecActive]    = useState(false)
  const [recSummary,   setRecSummary]   = useState(null)
  const [trainDone,    setTrainDone]    = useState(false)
  const [logLines,     setLogLines]     = useState([])
  const logRef = useRef(null)

  // Listen for WS non-inference messages
  useEffect(() => {
    if (!wsMessages.length) return
    const last = wsMessages[wsMessages.length - 1]

    if (last.type === 'user_loaded') {
      setStep('RECORD')
      addLog(`✅ Welcome back, ${last.username}! You have ${last.summary?.total_combos || 0} combos recorded.`)
      if (last.summary?.has_model) {
        addLog(`🧠 You already have a personal model (${last.summary?.combo_count || ''} combos)`)
      }
    }
    if (last.type === 'record_done') {
      setRecActive(false)
      setRecSummary(last)
      addLog(`✅ Session saved — ${last.combos?.length || 0} combos recorded this session`)
      addLog(`📦 Total combos in profile: ${last.total || 0}`)
      if ((last.total || 0) >= 5) setStep('TRAIN')
      else addLog(`⚠️  Record ${5 - (last.total||0)} more combos to unlock training`)
    }
    if (last.type === 'train_started') {
      addLog(`🏋️  Personal model training started for ${last.username}…`)
    }
    if (last.type === 'train_error') {
      addLog(`❌ Training error: ${last.msg}`)
    }
  }, [wsMessages])

  // Watch train progress
  useEffect(() => {
    if (!trainProgress) return
    if (trainProgress.msg) addLog(`[${trainProgress.pct}%] ${trainProgress.msg}`)
    if (trainProgress.pct === 100) {
      setTrainDone(true)
      setStep('READY')
      addLog('🎉 Your personal model is ready! Go to Free Play or Coach mode.')
    }
  }, [trainProgress])

  // Watch recording state from inference
  const recData = inference?.record

  function addLog(msg) {
    setLogLines(prev => [...prev.slice(-40), `[${new Date().toLocaleTimeString()}] ${msg}`])
    setTimeout(() => logRef.current?.scrollTo(0, 99999), 50)
  }

  // LOGIN step
  function handleLogin() {
    if (!username.trim()) { setInputErr('Enter a username'); return }
    if (!/^[a-z0-9_]{2,20}$/i.test(username.trim())) {
      setInputErr('2–20 letters/numbers/underscores only'); return
    }
    setInputErr('')
    loginUser(username.trim())
    addLog(`🔍 Loading profile for "${username.trim()}"…`)
    if (!cameraActive) startCamera()
  }

  // RECORD step
  async function handleStartRec() {
    if (!cameraActive) await startCamera()
    startRecording()
    setRecActive(true)
    addLog('🔴 Recording started — throw your natural combos. Pause 2s between each combo.')
  }

  function handleStopRec() {
    stopRecording()
    addLog('⏹  Recording stopped — saving combos…')
  }

  // TRAIN step
  function handleTrain() {
    trainPersonal()
    addLog('🚀 Sending combos to train your personal GRU model…')
  }

  const stepIdx   = STEPS.indexOf(step)
  const totalCombos = currentUser?.summary?.total_combos || 0
  const hasCam      = cameraActive && connected

  return (
    <div className={styles.page}>

      {/* ── LEFT: camera ── */}
      <div className={styles.camCol}>
        <div className={styles.colLabel}>▶ LIVE FEED</div>
        <div className={styles.camWrap}>
          <CameraFeed />
          {/* Live recording overlay */}
          {recActive && recData && (
            <div className={styles.recOverlay}>
              <div className={styles.recDot} />
              <div className={styles.recCombo}>
                {recData.current_combo?.length
                  ? recData.current_combo.join(' → ')
                  : 'Throw a punch…'}
              </div>
              <div className={styles.recSavedCount}>
                {recData.count} combo{recData.count !== 1 ? 's' : ''} saved
              </div>
            </div>
          )}
        </div>
        <div className={styles.camFooter}>
          <button
            className={`hex-btn ${cameraActive ? 'hex-btn-red' : 'hex-btn-green'}`}
            onClick={cameraActive ? stopCamera : startCamera}
          >
            {cameraActive ? '⬛ STOP CAMERA' : '▶ ACTIVATE CAMERA'}
          </button>
        </div>
      </div>

      {/* ── CENTER: steps ── */}
      <div className={styles.stepsCol}>

        {/* Progress bar */}
        <div className={styles.progressWrap}>
          {STEPS.map((s, i) => (
            <div key={s} className={`${styles.progressStep} ${i <= stepIdx ? styles.progressDone : ''}`}>
              <div className={styles.progressDot}>{i < stepIdx ? '✓' : i+1}</div>
              <div className={styles.progressLabel}>{s}</div>
              {i < STEPS.length-1 && <div className={styles.progressLine} />}
            </div>
          ))}
        </div>

        {/* ── STEP: LOGIN ── */}
        {step === 'LOGIN' && (
          <div className={`glass-panel ${styles.card}`}>
            <div className={styles.cardTitle}>◈ WHO ARE YOU?</div>
            <p className={styles.cardDesc}>
              Enter a username to load your profile. New users are created automatically.
              Each person trains their own personal AI model.
            </p>
            <div className={styles.inputRow}>
              <input
                className={styles.input}
                placeholder="your_username"
                value={username}
                onChange={e => setUsername(e.target.value)}
                onKeyDown={e => e.key==='Enter' && handleLogin()}
                spellCheck={false}
              />
              <button className="hex-btn hex-btn-blue" onClick={handleLogin}>
                LOAD →
              </button>
            </div>
            {inputErr && <div className={styles.err}>{inputErr}</div>}
            <div className={styles.hint}>Letters, numbers, underscores. 2–20 chars.</div>
          </div>
        )}

        {/* ── STEP: RECORD ── */}
        {step === 'RECORD' && (
          <div className={`glass-panel ${styles.card}`}>
            <div className={styles.cardTitle}>◈ RECORD YOUR COMBOS</div>
            <div className={styles.userBadge}>
              <span className={styles.userIcon}>👤</span>
              <span>{currentUser?.username}</span>
              <span className={styles.comboCount}>{totalCombos} combos saved</span>
            </div>
            <p className={styles.cardDesc}>
              Throw your natural combinations in front of the camera.
              Pause <strong>2 seconds</strong> between each combo — the AI will detect the gap
              and save it as one sequence. The more combos you record, the smarter your model.
            </p>

            <div className={styles.examplesBox}>
              <div className={styles.examplesLabel}>EXAMPLE COMBOS TO TRY</div>
              {[
                ['jab','cross'],
                ['jab','cross','hook'],
                ['jab','jab','cross'],
                ['hook','uppercut'],
                ['cross','hook','uppercut'],
                ['jab','cross','hook','uppercut'],
              ].map((c,i) => (
                <div key={i} className={styles.exampleRow}>
                  {c.map((m,j) => (
                    <span key={j}>
                      <span className={`${styles.moveChip} ${styles[m]}`}>{m.toUpperCase()}</span>
                      {j < c.length-1 && <span className={styles.arrow}>→</span>}
                    </span>
                  ))}
                </div>
              ))}
            </div>

            <div className={styles.btnRow}>
              {!recActive ? (
                <button className="hex-btn hex-btn-red" onClick={handleStartRec} disabled={!hasCam}>
                  🔴 START RECORDING
                </button>
              ) : (
                <button className="hex-btn hex-btn-blue" onClick={handleStopRec}>
                  ⏹ STOP & SAVE
                </button>
              )}
              {totalCombos >= 5 && !recActive && (
                <button className="hex-btn hex-btn-green" onClick={() => setStep('TRAIN')}>
                  NEXT: TRAIN MODEL →
                </button>
              )}
            </div>

            {totalCombos < 5 && (
              <div className={styles.needMore}>
                Need {5 - totalCombos} more combo{5-totalCombos!==1?'s':''} to unlock training
              </div>
            )}
          </div>
        )}

        {/* ── STEP: TRAIN ── */}
        {step === 'TRAIN' && (
          <div className={`glass-panel ${styles.card}`}>
            <div className={styles.cardTitle}>◈ TRAIN YOUR PERSONAL MODEL</div>
            <div className={styles.userBadge}>
              <span className={styles.userIcon}>👤</span>
              <span>{currentUser?.username}</span>
              <span className={styles.comboCount}>{totalCombos} combos</span>
            </div>
            <p className={styles.cardDesc}>
              Your AI will learn <strong>your</strong> specific combo patterns —
              not generic boxing. After training, it will predict your next move based on
              how YOU personally like to throw combinations.
            </p>

            {/* Combo preview */}
            {currentUser?.summary?.top_transitions?.length > 0 && (
              <div className={styles.transBox}>
                <div className={styles.examplesLabel}>YOUR TOP TRANSITIONS</div>
                {currentUser.summary.top_transitions.map((t,i) => (
                  <div key={i} className={styles.transRow}>
                    <span className={styles.transPair}>{t.pair}</span>
                    <div className={styles.transBar}>
                      <div className={styles.transBarFill}
                           style={{width:`${Math.min(100,(t.count/Math.max(...currentUser.summary.top_transitions.map(x=>x.count),1))*100)}%`}}/>
                    </div>
                    <span className={styles.transCount}>×{t.count}</span>
                  </div>
                ))}
              </div>
            )}

            {!trainProgress?.training && !trainDone && (
              <button className="hex-btn hex-btn-green" style={{width:'100%',justifyContent:'center'}}
                      onClick={handleTrain}>
                🧠 TRAIN MY PERSONAL MODEL
              </button>
            )}

            {/* Training progress */}
            {trainProgress?.training && (
              <div className={styles.progressBarWrap}>
                <div className={styles.progressBarLabel}>TRAINING… {trainProgress.pct}%</div>
                <div className={styles.progressBarTrack}>
                  <div className={styles.progressBarFill} style={{width:`${trainProgress.pct}%`}} />
                </div>
                <div className={styles.progressBarMsg}>{trainProgress.msg}</div>
              </div>
            )}

            {trainDone && (
              <div className={styles.trainDone}>
                🎉 Personal model trained! Head to Free Play or Coach Mode.
              </div>
            )}

            <button className={`${styles.backBtn} hex-btn hex-btn-blue`}
                    onClick={() => setStep('RECORD')}>
              ← RECORD MORE COMBOS
            </button>
          </div>
        )}

        {/* ── STEP: READY ── */}
        {step === 'READY' && (
          <div className={`glass-panel ${styles.card}`}>
            <div className={styles.cardTitle}>◈ YOUR MODEL IS READY</div>
            <div className={styles.readyIcon}>🥊</div>
            <p className={styles.cardDesc}>
              Your personal GRU model has been trained on your combo style.
              The AI will now predict YOUR next move — not generic boxing patterns.
            </p>
            <div className={styles.readyStats}>
              <div className={styles.readyStat}>
                <div className={styles.readyStatVal} style={{color:'var(--neon-green)'}}>{totalCombos}</div>
                <div className={styles.readyStatLabel}>COMBOS TRAINED</div>
              </div>
              <div className={styles.readyStat}>
                <div className={styles.readyStatVal} style={{color:'var(--neon-blue)'}}>
                  {currentUser?.summary?.total_punches || 0}
                </div>
                <div className={styles.readyStatLabel}>TOTAL PUNCHES</div>
              </div>
            </div>
            <button className="hex-btn hex-btn-green" style={{width:'100%',justifyContent:'center'}}
                    onClick={() => setStep('RECORD')}>
              + RECORD MORE COMBOS
            </button>
          </div>
        )}

      </div>

      {/* ── RIGHT: activity log ── */}
      <div className={styles.logCol}>
        <div className={styles.colLabel}>◈ ACTIVITY LOG</div>
        <div className={`glass-panel ${styles.logPanel}`}>
          <div className={styles.logLines} ref={logRef}>
            {logLines.length === 0
              ? <div className={styles.logEmpty}>Log is empty — start by entering a username.</div>
              : logLines.map((l,i) => <div key={i} className={styles.logLine}>{l}</div>)
            }
          </div>
        </div>

        {/* Recorded combos this session */}
        {recSummary?.combos?.length > 0 && (
          <div className={`glass-panel ${styles.savedPanel}`}>
            <div className="hud-label" style={{padding:'12px 14px 8px'}}>
              SAVED THIS SESSION ({recSummary.combos.length})
            </div>
            <div className={styles.savedList}>
              {recSummary.combos.map((c,i) => (
                <div key={i} className={styles.savedCombo}>
                  <span className={styles.savedIdx}>{String(i+1).padStart(2,'0')}</span>
                  {c.join(' → ')}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Move distribution */}
        {currentUser?.summary?.move_counts && (
          <div className={`glass-panel ${styles.distPanel}`}>
            <div className="hud-label" style={{padding:'12px 14px 8px'}}>YOUR MOVE USAGE</div>
            <div className={styles.distBars}>
              {Object.entries(currentUser.summary.move_counts).map(([m,v]) => {
                const total = Object.values(currentUser.summary.move_counts).reduce((a,b)=>a+b,0)
                const pct   = total ? Math.round(v/total*100) : 0
                const COL   = {jab:'#ffea00',cross:'#00dcff',hook:'#00e676',uppercut:'#d500f9'}
                return (
                  <div key={m} className={styles.distRow}>
                    <div className={styles.distName} style={{color:COL[m]}}>{m.toUpperCase()}</div>
                    <div className={styles.distTrack}>
                      <div className={styles.distFill} style={{width:`${pct}%`,background:COL[m]}}/>
                    </div>
                    <div className={styles.distPct}>{pct}%</div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useWS } from '../context/WSContext'
import styles from './Stats.module.css'

const MOVES  = ['jab','cross','hook','uppercut']
const COLORS = { jab:'#ffea00',cross:'#00dcff',hook:'#00e676',uppercut:'#d500f9' }

function Ring({ pct, color, size=80 }) {
  const r=( size-10)/2, circ=2*Math.PI*r, offset=circ-(pct/100)*circ
  return (
    <svg width={size} height={size}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6"/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth="6"
              strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
              transform={`rotate(-90 ${size/2} ${size/2})`}
              style={{filter:`drop-shadow(0 0 6px ${color})`,transition:'stroke-dashoffset 0.5s'}}/>
      <text x="50%" y="50%" textAnchor="middle" dy=".35em"
            fill={color} fontSize="14" fontFamily="var(--font-hud)" fontWeight="700">{pct}%</text>
    </svg>
  )
}

function MiniChart({ data, color }) {
  const max=Math.max(...data,1)
  return (
    <div className={styles.chartBars}>
      {data.map((v,i)=>(
        <div key={i} className={styles.chartBarCol}>
          <div className={styles.chartBar}
               style={{height:`${(v/max)*100}%`,background:color,boxShadow:`0 0 6px ${color}88`}}/>
        </div>
      ))}
    </div>
  )
}

export default function Stats() {
  const { inference, currentUser } = useWS()
  const nav = useNavigate()
  const [userData, setUserData] = useState(null)

  // Use currentUser summary from WS context (updated live)
  useEffect(() => {
    if (currentUser?.summary) setUserData(currentUser.summary)
  }, [currentUser])

  const liveCoach  = inference?.coach || null
  const liveActive = liveCoach?.active

  // Build charts from session history
  const sessions   = userData?.session_stats || []
  const accTrend   = sessions.map(s=>s.accuracy||0)
  const timeTrend  = sessions.map(s=>Math.round((s.avg_time||0)*100))
  const moveDist   = userData?.move_counts || {jab:0,cross:0,hook:0,uppercut:0}
  const totalMoves = Object.values(moveDist).reduce((a,b)=>a+b,0)||1
  const topTrans   = userData?.top_transitions || []

  return (
    <div className={styles.page}>

      {/* Header */}
      <div className={styles.topRow}>
        <div className={styles.pageTitleHud}>◉ PERFORMANCE ANALYTICS</div>
        {!currentUser && (
          <button className="hex-btn hex-btn-yellow" style={{fontSize:'9px',padding:'6px 16px'}}
                  onClick={()=>nav('/training')}>
            ↑ SET UP PROFILE TO SEE PERSONAL STATS
          </button>
        )}
        {liveActive && (
          <div className={styles.liveBadge}><div className={styles.liveDot}/>LIVE SESSION</div>
        )}
      </div>

      {/* Live stats (if coach running) */}
      {liveActive && liveCoach?.stats && (
        <div className={styles.liveRow}>
          {[
            {l:'LIVE REPS',   v:liveCoach.stats.correct,               c:'var(--neon-green)'},
            {l:'ACCURACY',    v:`${liveCoach.stats.accuracy||0}%`,       c:'var(--neon-yellow)'},
            {l:'AVG REACT',   v:`${liveCoach.stats.avg_time||0}s`,       c:'var(--neon-blue)'},
            {l:'MISTAKES',    v:liveCoach.stats.mistakes,                c:'var(--neon-red)'},
          ].map(({l,v,c})=>(
            <div key={l} className={`glass-panel ${styles.liveCard}`}>
              <div className="hud-label">{l}</div>
              <div className={styles.liveVal} style={{color:c}}>{v}</div>
            </div>
          ))}
        </div>
      )}

      <div className={styles.mainGrid}>

        {/* Session history */}
        <div className={`glass-panel ${styles.histPanel}`}>
          <div className="hud-label" style={{padding:'14px 18px 10px'}}>
            {userData ? `${userData.username?.toUpperCase()} — SESSION HISTORY` : 'SESSION HISTORY'}
          </div>
          {sessions.length===0 ? (
            <div className={styles.noData}>
              {currentUser
                ? 'No sessions yet — complete a Coach Mode session to see history.'
                : 'Set up your profile in Training to track personal sessions.'}
            </div>
          ) : (
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead><tr><th>#</th><th>DATE</th><th>REPS</th><th>ACC</th><th>TIME</th><th>ERR</th></tr></thead>
                <tbody>
                  {sessions.slice().reverse().map((s,i)=>(
                    <tr key={i}>
                      <td className={styles.tdNum}>{String(sessions.length-i).padStart(2,'0')}</td>
                      <td>{s.date}</td>
                      <td style={{fontFamily:'var(--font-hud)',fontSize:'11px',color:'var(--neon-blue)'}}>{s.correct}</td>
                      <td style={{fontFamily:'var(--font-hud)',fontSize:'12px',
                                  color:s.accuracy>=90?'var(--neon-green)':s.accuracy>=70?'var(--neon-yellow)':'var(--neon-red)'}}>
                        {s.accuracy}%</td>
                      <td style={{fontFamily:'var(--font-hud)',fontSize:'11px',color:'var(--neon-cyan)'}}>{s.avg_time}s</td>
                      <td style={{fontFamily:'var(--font-hud)',fontSize:'11px',
                                  color:s.mistakes>0?'var(--neon-red)':'var(--neon-green)'}}>{s.mistakes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Charts */}
        <div className={styles.chartsCol}>
          <div className={`glass-panel ${styles.chartCard}`}>
            <div className="hud-label" style={{padding:'12px 16px 8px'}}>ACCURACY TREND</div>
            {accTrend.length
              ? <MiniChart data={accTrend} color="var(--neon-green)"/>
              : <div className={styles.noDataSm}>No sessions yet</div>}
          </div>
          <div className={`glass-panel ${styles.chartCard}`}>
            <div className="hud-label" style={{padding:'12px 16px 8px'}}>REACTION TIME (×10ms)</div>
            {timeTrend.length
              ? <MiniChart data={timeTrend} color="var(--neon-blue)"/>
              : <div className={styles.noDataSm}>No sessions yet</div>}
          </div>
        </div>

        {/* Distribution + Top transitions */}
        <div className={styles.distPanel}>
          <div className={`glass-panel ${styles.distCard}`}>
            <div className="hud-label" style={{padding:'14px 18px 12px'}}>YOUR MOVE USAGE</div>
            <div className={styles.distGrid}>
              {MOVES.map(m=>(
                <div key={m} className={styles.distItem}>
                  <Ring pct={totalMoves?Math.round(moveDist[m]/totalMoves*100):0} color={COLORS[m]} size={72}/>
                  <div className={styles.distName} style={{color:COLORS[m]}}>{m.toUpperCase()}</div>
                </div>
              ))}
            </div>
          </div>

          {topTrans.length>0 && (
            <div className={`glass-panel ${styles.transCard}`}>
              <div className="hud-label" style={{padding:'12px 16px 10px'}}>YOUR TOP TRANSITIONS</div>
              {topTrans.map((t,i)=>(
                <div key={i} className={styles.transRow}>
                  <span className={styles.transPair}>{t.pair}</span>
                  <div className={styles.transBar}>
                    <div className={styles.transBarFill}
                         style={{width:`${Math.round(t.count/Math.max(...topTrans.map(x=>x.count),1)*100)}%`}}/>
                  </div>
                  <span className={styles.transCount}>×{t.count}</span>
                </div>
              ))}
            </div>
          )}

          <div className={`glass-panel ${styles.totalCard}`}>
            <div className="hud-label" style={{padding:'12px 16px 8px'}}>PROFILE TOTALS</div>
            <div className={styles.totalGrid}>
              <div className={styles.totalItem}>
                <div className={styles.totalVal} style={{color:'var(--neon-blue)'}}>{userData?.total_combos||0}</div>
                <div className={styles.totalLabel}>COMBOS RECORDED</div>
              </div>
              <div className={styles.totalItem}>
                <div className={styles.totalVal} style={{color:'var(--neon-green)'}}>{userData?.total_punches||0}</div>
                <div className={styles.totalLabel}>TOTAL PUNCHES</div>
              </div>
              <div className={styles.totalItem}>
                <div className={styles.totalVal} style={{color:'var(--neon-yellow)'}}>{userData?.total_sessions||0}</div>
                <div className={styles.totalLabel}>SESSIONS</div>
              </div>
              <div className={styles.totalItem}>
                <div className={styles.totalVal} style={{color:userData?.has_model?'var(--neon-green)':'var(--neon-red)'}}>
                  {userData?.has_model?'YES':'NO'}
                </div>
                <div className={styles.totalLabel}>PERSONAL MODEL</div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
import { useLocation, useNavigate } from 'react-router-dom'
import { useWS } from '../context/WSContext'
import styles from './Navbar.module.css'

const NAV_ITEMS = [
  { path:'/',          label:'HOME',     icon:'◈' },
  { path:'/training',  label:'TRAINING', icon:'🧠', highlight: true },
  { path:'/freeplay',  label:'FREE PLAY',icon:'⚡' },
  { path:'/coach',     label:'COACH',    icon:'⚔' },
  { path:'/stats',     label:'STATS',    icon:'◉' },
]

export default function Navbar() {
  const { pathname } = useLocation()
  const nav = useNavigate()
  const { connected, cameraActive, currentUser } = useWS()

  return (
    <nav className={styles.nav}>
      <div className={styles.logo} onClick={() => nav('/')}>
        <span className={styles.logoIcon}>⚡</span>
        <span className={styles.logoText}>BOXING<em>AI</em></span>
      </div>

      <div className={styles.links}>
        {NAV_ITEMS.map(item => (
          <button
            key={item.path}
            className={`${styles.link} ${pathname===item.path ? styles.active:''} ${item.highlight?styles.highlight:''}`}
            onClick={() => nav(item.path)}
          >
            <span className={styles.linkIcon}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </div>

      <div className={styles.right}>
        {/* User badge */}
        {currentUser ? (
          <div className={styles.userBadge} onClick={() => nav('/training')}>
            <span className={styles.userIcon}>👤</span>
            <span className={styles.userName}>{currentUser.username}</span>
            {currentUser.summary?.has_model && (
              <span className={styles.modelBadge}>✓ MODEL</span>
            )}
          </div>
        ) : (
          <button className={styles.loginHint} onClick={() => nav('/training')}>
            + SET UP PROFILE
          </button>
        )}

        <div className={`${styles.dot} ${connected?styles.dotLive:''}`} />
        <span className={styles.statusText}>
          {connected ? 'LIVE' : 'OFFLINE'}
        </span>
      </div>
    </nav>
  )
}
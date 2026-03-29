import { useRef, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useScroll, useTransform, motion } from 'framer-motion'
import { Brain, Zap, Swords, BarChart2 } from 'lucide-react'
import { useWS } from '../context/WSContext'
import TechniqueAnimator from '../components/TechniqueAnimator'
import RadialOrbitalTimeline from '../components/RadialOrbitalTimeline'
import styles from './Home.module.css'

// ── Feature cards data ────────────────────────────────────────
const CARDS = [
  {
    id: 1, path: '/training', icon: Brain, title: 'SETUP & TRAIN',
    sub: 'Create your profile, record your personal combos, and train your own AI prediction model that learns YOUR style.',
    content: 'Record combos, label them, and train a personal TCN + GRU model that predicts your next move in real-time.',
    color: '#ffea00', tag: 'START HERE', num: '01',
    date: 'STEP 1', relatedIds: [2], status: 'completed', energy: 100,
  },
  {
    id: 2, path: '/freeplay', icon: Zap, title: 'FREE PLAY',
    sub: 'Shadow box freely. Real-time punch recognition with your personal next-move prediction running at 25 FPS.',
    content: 'MediaPipe pose → TCN recognition → GRU prediction. Your trained model runs at 25 FPS with locked confirmations.',
    color: '#00dcff', tag: 'LIVE AI', num: '02',
    date: 'STEP 2', relatedIds: [1, 3], status: 'in-progress', energy: 80,
  },
  {
    id: 3, path: '/coach', icon: Swords, title: 'COACH MODE',
    sub: 'Follow commands. The AI grades your reaction time and accuracy against the moves you were trained on.',
    content: 'Coach calls a move, you execute it. AI grades reaction time and accuracy against your personal style.',
    color: '#ff1744', tag: 'TRAINING', num: '03',
    date: 'STEP 3', relatedIds: [2, 4], status: 'in-progress', energy: 65,
  },
  {
    id: 4, path: '/stats', icon: BarChart2, title: 'ANALYTICS',
    sub: 'Review your performance history, combo patterns, accuracy trends, and improvement over every session.',
    content: 'Session history, accuracy trends, combo heatmaps, and model improvement tracking across every training run.',
    color: '#00e676', tag: 'STATS', num: '04',
    date: 'STEP 4', relatedIds: [3], status: 'pending', energy: 40,
  },
]

// ── Hover button ──────────────────────────────────────────────
function HoverButton({ text, onClick, color = 'var(--neon-blue)', secondary = false }) {
  return (
    <button
      className={`${styles.hoverBtn} ${secondary ? styles.hoverBtnSecondary : ''}`}
      style={{ '--btn-color': color }}
      onClick={onClick}
    >
      <span className={styles.hoverBtnText}>{text}</span>
      <span className={styles.hoverBtnReveal}>
        {text} <span className={styles.hoverBtnArrow}>→</span>
      </span>
      <span className={styles.hoverBtnBg} />
    </button>
  )
}

// ── ContainerScroll ───────────────────────────────────────────
function ContainerScroll({ titleComponent, children }) {
  const containerRef = useRef(null)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth <= 768)
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  const { scrollYProgress } = useScroll({ target: containerRef })
  const rotate = useTransform(scrollYProgress, [0, 1], [20, 0])
  const scale = useTransform(scrollYProgress, [0, 1], isMobile ? [0.7, 0.9] : [1.05, 1])
  const translate = useTransform(scrollYProgress, [0, 1], [0, -100])

  return (
    <div ref={containerRef}
      className={isMobile ? styles.scrollContainer : styles.scrollContainerDesktop}>
      <div className={styles.scrollInner}>
        <motion.div className={styles.scrollTitleWrap} style={{ translateY: translate }}>
          {titleComponent}
        </motion.div>
        <motion.div
          className={styles.scrollCard}
          style={{
            rotateX: rotate, scale,
            boxShadow: [
              '0 0 #0000004d', '0 9px 20px #0000004a', '0 37px 37px #00000042',
              '0 84px 50px #00000026', '0 149px 60px #0000000a', '0 233px 65px #00000003',
            ].join(', '),
          }}
        >
          <div className={styles.scrollCardInner}>{children}</div>
        </motion.div>
      </div>
    </div>
  )
}

// ── Main Home page ────────────────────────────────────────────
export default function Home() {
  const nav = useNavigate()
  const { currentUser } = useWS()

  return (
    <div className={styles.page}>

      {/* ══════════════════════════════════════════════════
          SECTION 1 — HERO
          Left: title + user banner + nav buttons
          Right: TechniqueAnimator in demo mode
                 (auto-cycles jab → cross → hook → uppercut)
          Zero 3rd-party scene — pure canvas, no lag
      ══════════════════════════════════════════════════ */}
      <section className={styles.hero}>
        <div className={styles.heroGrid} aria-hidden="true" />

        <div className={styles.heroInner}>

          {/* LEFT ─────────────────────────────────────── */}
          <div className={styles.heroLeft}>
            <div className={styles.heroBadge}>NEURAL COMBAT SYSTEM v3.0</div>

            <h1 className={styles.heroTitle}>
              <span className={styles.heroTitleBoxing}>BOXING</span>
              <span className={styles.heroTitleAI}>AI</span>
            </h1>

            <p className={styles.heroSubtitle}>
              Record your combos · Train your personal model<br />
              Predict YOUR next move in real-time
            </p>

            {currentUser ? (
              <div className={styles.userBanner}>
                <span>👤</span>
                &nbsp;Welcome back, <strong>{currentUser.username}</strong>
                {currentUser.summary?.has_model
                  ? <span className={styles.modelReady}> · Model ready ✓</span>
                  : <span className={styles.modelNeeded}
                    onClick={() => nav('/training')}> · Build your model →</span>
                }
              </div>
            ) : (
              <div className={styles.newUserBanner} onClick={() => nav('/training')}>
                ⚡ New here? Start with Setup &amp; Train
              </div>
            )}

            <div className={styles.heroBtns}>
              <HoverButton text="START TRAINING" onClick={() => nav('/training')} color="var(--neon-yellow)" />
              <HoverButton text="FREE PLAY" onClick={() => nav('/freeplay')} color="var(--neon-blue)" secondary />
            </div>

            <div className={styles.heroHint}>
              SPACE — camera &nbsp;·&nbsp; F — free play &nbsp;·&nbsp; C — coach
            </div>
          </div>

          {/* RIGHT — demo animator ───────────────────── */}
          <div className={styles.heroRight}>
            <div className={styles.heroAnimCornerTL} />
            <div className={styles.heroAnimCornerTR} />
            <div className={styles.heroAnimCornerBL} />
            <div className={styles.heroAnimCornerBR} />
            <div className={styles.heroAnimLabel}>◈ TECHNIQUE PREVIEW</div>
            <TechniqueAnimator move="demo" />
          </div>

        </div>

        <div className={styles.heroScrollHint}>
          <div className={styles.heroScrollDot} />
          SCROLL TO EXPLORE
        </div>
      </section>

      {/* ══════════════════════════════════════════════════
          SECTION 2 — CONTAINERSCROLL
      ══════════════════════════════════════════════════ */}
      <section className={styles.scrollSection}>
        <ContainerScroll
          titleComponent={
            <div className={styles.scrollTitleBlock}>
              <div className={styles.scrollBadge}>HOW IT WORKS</div>
              <h2 className={styles.scrollHeading}>
                YOUR PERSONAL<br />
                <span className={styles.scrollHeadingAccent}>BOXING AI</span>
              </h2>
              <p className={styles.scrollSubtitle}>
                Record combos → Train your model → Real-time prediction
              </p>
            </div>
          }
        >
          <div className={styles.previewCols}>
            <div className={styles.previewCol}>
              <div className={styles.previewLabel}>LIVE CAMERA</div>
              <div className={styles.previewCam}>
                <div className={styles.previewCamInner}>
                  <span className={styles.previewCamIcon}>📷</span>
                  <span className={styles.previewCamText}>POSE LOCKED</span>
                </div>
              </div>
              <div className={styles.previewBtn} />
            </div>
            <div className={styles.previewCol}>
              <div className={styles.previewLabel}>AI DETECTION</div>
              <div className={styles.previewHud}>
                {['JAB', 'CROSS', 'HOOK', 'UPPERCUT'].map((m, i) => (
                  <div key={m} className={styles.previewBar}>
                    <span className={styles.previewBarName}>{m}</span>
                    <div className={styles.previewBarTrack}>
                      <div className={styles.previewBarFill}
                        style={{
                          width: ['78%', '15%', '5%', '2%'][i],
                          background: ['#ffea00', '#00dcff', '#00e676', '#d500f9'][i],
                        }}
                      />
                    </div>
                  </div>
                ))}
                <div className={styles.previewMoveBox}>JAB</div>
              </div>
            </div>
            <div className={styles.previewCol}>
              <div className={styles.previewLabel}>TECHNIQUE</div>
              <div className={styles.previewAnimBox}>
                <div className={styles.previewAnimIcon}>🥊</div>
                <div className={styles.previewAnimText}>ANIMATION</div>
              </div>
            </div>
          </div>
        </ContainerScroll>
      </section>

      {/* ══════════════════════════════════════════════════
          SECTION 3 — ORBITAL TIMELINE
      ══════════════════════════════════════════════════ */}
      <section className={styles.cardsSection}>
        <div className={styles.cardsSectionHeader}>
          <div className={styles.cardsBadge}>WHAT YOU CAN DO</div>
          <h2 className={styles.cardsHeading}>CHOOSE YOUR MODE</h2>
          <p className={styles.cardsSubtitle}>Click any node to explore — click background to rotate</p>
        </div>
        <RadialOrbitalTimeline
          timelineData={CARDS}
          onNavigate={path => nav(path)}
        />
      </section>

      {/* ══════════════════════════════════════════════════
          SECTION 4 — CTA
      ══════════════════════════════════════════════════ */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaGlow} aria-hidden="true" />
        <div className={styles.ctaContent}>
          <div className={styles.ctaBadge}>GET STARTED</div>
          <h2 className={styles.ctaHeading}>READY TO TRAIN?</h2>
          <p className={styles.ctaSubtitle}>Your personal boxing AI is one recording session away</p>
          <div className={styles.ctaBtns}>
            <HoverButton text="BEGIN TRAINING" onClick={() => nav('/training')} color="var(--neon-red)" />
            <HoverButton text="VIEW STATS" onClick={() => nav('/stats')} color="var(--neon-green)" secondary />
          </div>
          <div className={styles.ctaHint}>No setup required — open in browser, allow camera, start punching</div>
        </div>
      </section>

    </div>
  )
}
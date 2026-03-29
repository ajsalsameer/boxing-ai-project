/**
 * RadialOrbitalTimeline
 * Translated from the reference: TypeScript → JSX, Tailwind → inline styles, shadcn → plain HTML
 *
 * Auto-rotates. Click a node to expand its detail card and pause rotation.
 * Click the background or orbit ring to resume.
 */

import { useState, useEffect, useRef } from 'react'
import { ArrowRight, Zap, Link2 } from 'lucide-react'

// ── Shared inline style tokens matching the neon-noir design system ──
const T = {
    fontHud: "'Orbitron', monospace",
    fontBody: "'Rajdhani', sans-serif",
    neonBlue: '#00dcff',
    bg: '#04040a',
}

// ── Status badge ─────────────────────────────────────────────
function StatusBadge({ status }) {
    const map = {
        completed: { label: 'COMPLETE', bg: 'rgba(0,230,118,0.15)', color: '#00e676', border: 'rgba(0,230,118,0.4)' },
        'in-progress': { label: 'IN PROGRESS', bg: 'rgba(0,220,255,0.15)', color: '#00dcff', border: 'rgba(0,220,255,0.4)' },
        pending: { label: 'PENDING', bg: 'rgba(255,234,0,0.1)', color: '#ffea00', border: 'rgba(255,234,0,0.3)' },
    }
    const s = map[status] || map.pending
    return (
        <span style={{
            fontFamily: T.fontHud, fontSize: 7, letterSpacing: 2,
            padding: '3px 8px', border: `1px solid ${s.border}`,
            background: s.bg, color: s.color, whiteSpace: 'nowrap',
        }}>
            {s.label}
        </span>
    )
}

// ── Expanded node card ────────────────────────────────────────
function NodeCard({ item, onNavigate, onRelatedClick }) {
    return (
        <div style={{
            position: 'absolute',
            top: 52, left: '50%',
            transform: 'translateX(-50%)',
            width: 230,
            background: 'rgba(4,4,10,0.96)',
            border: '1px solid rgba(255,255,255,0.15)',
            backdropFilter: 'blur(16px)',
            boxShadow: '0 8px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.04)',
            zIndex: 300,
            overflow: 'visible',
        }}>
            {/* Connector line from node to card */}
            <div style={{
                position: 'absolute', top: -12, left: '50%', transform: 'translateX(-50%)',
                width: 1, height: 12, background: 'rgba(255,255,255,0.25)',
            }} />

            {/* Card header */}
            <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <StatusBadge status={item.status} />
                    <span style={{ fontFamily: T.fontHud, fontSize: 7, color: 'rgba(255,255,255,0.35)', letterSpacing: 1 }}>
                        {item.date}
                    </span>
                </div>
                <div style={{ fontFamily: T.fontHud, fontSize: 11, fontWeight: 700, letterSpacing: 2, color: '#fff' }}>
                    {item.title}
                </div>
            </div>

            {/* Card body */}
            <div style={{ padding: '10px 14px 12px' }}>
                <p style={{ fontFamily: T.fontBody, fontSize: 11, color: 'rgba(204,214,246,0.6)', lineHeight: 1.6, marginBottom: 12 }}>
                    {item.content}
                </p>

                {/* Energy bar */}
                <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span style={{ fontFamily: T.fontBody, fontSize: 10, color: 'rgba(255,255,255,0.45)', display: 'flex', alignItems: 'center', gap: 4 }}>
                            <Zap size={9} /> Energy
                        </span>
                        <span style={{ fontFamily: T.fontHud, fontSize: 9, color: 'rgba(255,255,255,0.5)' }}>{item.energy}%</span>
                    </div>
                    <div style={{ height: 3, background: 'rgba(255,255,255,0.08)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{
                            height: '100%', width: `${item.energy}%`,
                            background: `linear-gradient(90deg, ${item.color}, ${item.color}88)`,
                            borderRadius: 2,
                        }} />
                    </div>
                </div>

                {/* Connected nodes */}
                {item.relatedIds.length > 0 && (
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 6 }}>
                            <Link2 size={9} color="rgba(255,255,255,0.4)" />
                            <span style={{ fontFamily: T.fontHud, fontSize: 7, letterSpacing: 2, color: 'rgba(255,255,255,0.4)' }}>
                                CONNECTED
                            </span>
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {item.relatedIds.map(relId => (
                                <button
                                    key={relId}
                                    onClick={e => { e.stopPropagation(); onRelatedClick(relId) }}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: 4,
                                        fontFamily: T.fontHud, fontSize: 7, letterSpacing: 1,
                                        color: 'rgba(255,255,255,0.6)',
                                        background: 'transparent',
                                        border: '1px solid rgba(255,255,255,0.15)',
                                        padding: '3px 8px', cursor: 'pointer',
                                        transition: 'all 0.2s',
                                    }}
                                    onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = '#fff' }}
                                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'rgba(255,255,255,0.6)' }}
                                >
                                    {relId} <ArrowRight size={7} />
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Navigate button */}
                <button
                    onClick={e => { e.stopPropagation(); onNavigate() }}
                    style={{
                        marginTop: 12, width: '100%',
                        fontFamily: T.fontHud, fontSize: 8, letterSpacing: 3, fontWeight: 700,
                        color: item.color,
                        background: 'transparent',
                        border: `1px solid ${item.color}55`,
                        padding: '8px 0', cursor: 'pointer',
                        transition: 'all 0.2s',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = `${item.color}18`; e.currentTarget.style.borderColor = item.color }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = `${item.color}55` }}
                >
                    ENTER <ArrowRight size={10} />
                </button>
            </div>
        </div>
    )
}

// ── Main component ────────────────────────────────────────────
export default function RadialOrbitalTimeline({ timelineData, onNavigate }) {
    const [expandedId, setExpandedId] = useState(null)
    const [rotationAngle, setRotationAngle] = useState(0)
    const [autoRotate, setAutoRotate] = useState(true)
    const [pulseIds, setPulseIds] = useState({})
    const containerRef = useRef(null)
    const orbitRef = useRef(null)
    const timerRef = useRef(null)

    // Auto-rotation loop
    useEffect(() => {
        if (autoRotate) {
            timerRef.current = setInterval(() => {
                setRotationAngle(prev => +(((prev + 0.3) % 360).toFixed(3)))
            }, 50)
        }
        return () => clearInterval(timerRef.current)
    }, [autoRotate])

    // Click background → reset
    const handleBgClick = e => {
        if (e.target === containerRef.current || e.target === orbitRef.current) {
            setExpandedId(null)
            setAutoRotate(true)
            setPulseIds({})
        }
    }

    const toggleNode = id => {
        if (expandedId === id) {
            setExpandedId(null)
            setAutoRotate(true)
            setPulseIds({})
        } else {
            setExpandedId(id)
            setAutoRotate(false)
            // Snap orbit so selected node is at the front (bottom of orbit = 270°)
            const idx = timelineData.findIndex(t => t.id === id)
            const total = timelineData.length
            const target = (idx / total) * 360
            setRotationAngle(270 - target)
            // Pulse related nodes
            const item = timelineData.find(t => t.id === id)
            const pulse = {}
            if (item) item.relatedIds.forEach(rid => { pulse[rid] = true })
            setPulseIds(pulse)
        }
    }

    // Position each node on the orbit circle
    const nodePos = (index, total) => {
        const angle = ((index / total) * 360 + rotationAngle) % 360
        const rad = (angle * Math.PI) / 180
        const radius = 170
        const x = radius * Math.cos(rad)
        const y = radius * Math.sin(rad)
        // Nodes at the back are dimmer and lower z
        const zIndex = Math.round(100 + 50 * Math.cos(rad))
        const opacity = Math.max(0.35, Math.min(1, 0.35 + 0.65 * ((1 + Math.sin(rad)) / 2)))
        return { x, y, zIndex, opacity }
    }

    return (
        <div
            ref={containerRef}
            onClick={handleBgClick}
            style={{
                width: '100%', height: 520,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                position: 'relative', overflow: 'hidden',
                background: 'radial-gradient(ellipse at center, rgba(0,220,255,0.03) 0%, transparent 70%)',
            }}
        >
            {/* Orbit ring */}
            <div
                ref={orbitRef}
                style={{
                    position: 'absolute',
                    width: '100%', height: '100%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
            >
                {/* Outer ring */}
                <div style={{
                    position: 'absolute',
                    width: 340, height: 340,
                    borderRadius: '50%',
                    border: '1px solid rgba(0,220,255,0.08)',
                    pointerEvents: 'none',
                }} />
                {/* Subtle inner ring */}
                <div style={{
                    position: 'absolute',
                    width: 100, height: 100,
                    borderRadius: '50%',
                    border: '1px solid rgba(0,220,255,0.06)',
                    pointerEvents: 'none',
                }} />

                {/* ── Center orb ── */}
                <div style={{
                    position: 'absolute', zIndex: 10,
                    width: 56, height: 56, borderRadius: '50%',
                    background: 'linear-gradient(135deg, #7c3aed, #2563eb, #0d9488)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    animation: 'orbPulse 2.5s ease-in-out infinite',
                    boxShadow: '0 0 30px rgba(99,102,241,0.5), 0 0 60px rgba(99,102,241,0.2)',
                }}>
                    {/* Ping rings */}
                    <div style={{
                        position: 'absolute', width: 72, height: 72, borderRadius: '50%',
                        border: '1px solid rgba(255,255,255,0.15)',
                        animation: 'orbPing 1.2s cubic-bezier(0,0,0.2,1) infinite',
                    }} />
                    <div style={{
                        position: 'absolute', width: 88, height: 88, borderRadius: '50%',
                        border: '1px solid rgba(255,255,255,0.08)',
                        animation: 'orbPing 1.2s cubic-bezier(0,0,0.2,1) infinite',
                        animationDelay: '0.4s',
                    }} />
                    {/* Core white dot */}
                    <div style={{
                        width: 24, height: 24, borderRadius: '50%',
                        background: 'rgba(255,255,255,0.85)',
                        backdropFilter: 'blur(4px)',
                    }} />
                </div>

                {/* ── Orbit nodes ── */}
                {timelineData.map((item, index) => {
                    const pos = nodePos(index, timelineData.length)
                    const isExpanded = expandedId === item.id
                    const isRelated = !isExpanded && expandedId && timelineData
                        .find(t => t.id === expandedId)?.relatedIds.includes(item.id)
                    const isPulsing = pulseIds[item.id]
                    const Icon = item.icon

                    return (
                        <div
                            key={item.id}
                            onClick={e => { e.stopPropagation(); toggleNode(item.id) }}
                            style={{
                                position: 'absolute',
                                transform: `translate(${pos.x}px, ${pos.y}px)`,
                                zIndex: isExpanded ? 250 : pos.zIndex,
                                opacity: isExpanded ? 1 : pos.opacity,
                                transition: 'opacity 0.4s, transform 0.1s',
                                cursor: 'pointer',
                            }}
                        >
                            {/* Energy aura */}
                            <div style={{
                                position: 'absolute',
                                width: item.energy * 0.35 + 40,
                                height: item.energy * 0.35 + 40,
                                left: -(item.energy * 0.35) / 2,
                                top: -(item.energy * 0.35) / 2,
                                borderRadius: '50%',
                                background: `radial-gradient(circle, ${item.color}22 0%, transparent 70%)`,
                                animation: isPulsing ? 'orbPulse 1s ease-in-out infinite' : 'none',
                            }} />

                            {/* Node circle */}
                            <div style={{
                                width: 40, height: 40, borderRadius: '50%',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: isExpanded ? item.color
                                    : isRelated ? `${item.color}55`
                                        : 'rgba(4,4,10,0.9)',
                                border: `2px solid ${isExpanded ? item.color
                                        : isRelated ? item.color
                                            : 'rgba(255,255,255,0.25)'
                                    }`,
                                boxShadow: isExpanded ? `0 0 20px ${item.color}88, 0 0 40px ${item.color}33` : 'none',
                                transform: isExpanded ? 'scale(1.45)' : 'scale(1)',
                                transition: 'all 0.3s ease',
                                color: isExpanded ? '#000' : '#fff',
                                animation: isRelated ? 'orbPulse 1s ease-in-out infinite' : 'none',
                            }}>
                                <Icon size={14} />
                            </div>

                            {/* Node label */}
                            <div style={{
                                position: 'absolute', top: 46,
                                left: '50%', transform: 'translateX(-50%)',
                                whiteSpace: 'nowrap',
                                fontFamily: T.fontHud, fontSize: 8, letterSpacing: 2,
                                color: isExpanded ? item.color : 'rgba(255,255,255,0.55)',
                                textShadow: isExpanded ? `0 0 12px ${item.color}` : 'none',
                                transition: 'all 0.3s',
                            }}>
                                {item.title}
                            </div>

                            {/* Expanded detail card */}
                            {isExpanded && (
                                <NodeCard
                                    item={item}
                                    onNavigate={() => onNavigate(item.path)}
                                    onRelatedClick={id => toggleNode(id)}
                                />
                            )}
                        </div>
                    )
                })}
            </div>

            {/* Keyframes injected once */}
            <style>{`
        @keyframes orbPulse {
          0%,100% { opacity:1; transform:scale(1); }
          50%      { opacity:0.7; transform:scale(1.08); }
        }
        @keyframes orbPing {
          75%,100% { transform:scale(1.6); opacity:0; }
        }
      `}</style>
        </div>
    )
}
/**
 * Spotlight component — two variants adapted from the reference docs
 *
 * 1. <SVGSpotlight />    — aceternity version: animated SVG beam that fades in
 * 2. <MouseSpotlight />  — ibelick version: radial glow that follows the mouse
 *
 * Both are translated from TypeScript → JSX, Tailwind → CSS-in-JS,
 * and use framer-motion (already in this project).
 */

import { useRef, useState, useCallback, useEffect } from 'react'
import { motion, useSpring, useTransform } from 'framer-motion'

// ── 1. SVG Spotlight beam (aceternity style) ────────────────
// Renders a large diagonal elliptical beam that animates in with opacity.
// `fill`  — beam colour  (default white)
// `style` — position overrides so it can be placed anywhere in the parent
export function SVGSpotlight({ fill = 'white', style = {} }) {
    return (
        <svg
            style={{
                position: 'absolute',
                zIndex: 1,
                pointerEvents: 'none',
                width: '138%',
                height: '169%',
                opacity: 0,
                animation: 'spotlightFadeIn 1.2s 0.3s ease forwards',
                ...style,
            }}
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 3787 2842"
            fill="none"
        >
            <g filter="url(#svgSpotFilter)">
                <ellipse
                    cx="1924.71"
                    cy="273.501"
                    rx="1924.71"
                    ry="273.501"
                    transform="matrix(-0.822377 -0.568943 -0.568943 0.822377 3631.88 2291.09)"
                    fill={fill}
                    fillOpacity="0.21"
                />
            </g>
            <defs>
                <filter
                    id="svgSpotFilter"
                    x="0.860352" y="0.838989"
                    width="3785.16" height="2840.26"
                    filterUnits="userSpaceOnUse"
                    colorInterpolationFilters="sRGB"
                >
                    <feFlood floodOpacity="0" result="BackgroundImageFix" />
                    <feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
                    <feGaussianBlur stdDeviation="151" result="effect1_foregroundBlur_1065_8" />
                </filter>
            </defs>
        </svg>
    )
}

// ── 2. Mouse-tracking spotlight (ibelick style) ─────────────
// A blurred radial circle that springs to follow the mouse cursor.
// Must be placed inside a `position: relative` parent.
// `size`  — diameter in px  (default 300)
// `color` — centre colour   (default white)
export function MouseSpotlight({
    size = 300,
    color = 'rgba(255,255,255,0.07)',
    springOpts = { bounce: 0, stiffness: 120, damping: 20 },
}) {
    const divRef = useRef(null)
    const [hovered, setHovered] = useState(false)
    const [parent, setParent] = useState(null)

    const mouseX = useSpring(0, springOpts)
    const mouseY = useSpring(0, springOpts)

    const left = useTransform(mouseX, x => `${x - size / 2}px`)
    const top = useTransform(mouseY, y => `${y - size / 2}px`)

    // Attach to the closest positioned parent
    useEffect(() => {
        if (divRef.current) {
            const p = divRef.current.parentElement
            if (p) {
                p.style.position = 'relative'
                p.style.overflow = 'hidden'
                setParent(p)
            }
        }
    }, [])

    const onMove = useCallback((e) => {
        if (!parent) return
        const { left: pl, top: pt } = parent.getBoundingClientRect()
        mouseX.set(e.clientX - pl)
        mouseY.set(e.clientY - pt)
    }, [parent, mouseX, mouseY])

    useEffect(() => {
        if (!parent) return
        parent.addEventListener('mousemove', onMove)
        parent.addEventListener('mouseenter', () => setHovered(true))
        parent.addEventListener('mouseleave', () => setHovered(false))
        return () => {
            parent.removeEventListener('mousemove', onMove)
            parent.removeEventListener('mouseenter', () => setHovered(true))
            parent.removeEventListener('mouseleave', () => setHovered(false))
        }
    }, [parent, onMove])

    return (
        <motion.div
            ref={divRef}
            style={{
                position: 'absolute',
                pointerEvents: 'none',
                borderRadius: '50%',
                width: size,
                height: size,
                left,
                top,
                background: `radial-gradient(circle at center, ${color} 0%, transparent 80%)`,
                filter: 'blur(32px)',
                opacity: hovered ? 1 : 0,
                transition: 'opacity 0.25s ease',
                zIndex: 2,
            }}
        />
    )
}
/**
 * SplineScene — lazy-loaded Spline 3D embed
 * Translated from the reference: TypeScript → JSX, 'use client' removed (not Next.js)
 *
 * Uses React.lazy + Suspense so the heavy @splinetool/react-spline bundle
 * only loads when the component is actually rendered, keeping initial page load fast.
 *
 * Props:
 *   scene     — spline .splinecode URL (required)
 *   className — extra class if needed (optional)
 *   style     — inline style overrides (optional)
 */

import { Suspense, lazy } from 'react'

const Spline = lazy(() => import('@splinetool/react-spline'))

// ── Loader shown while the 3D scene downloads ────────────────
function SplineLoader() {
    return (
        <div style={{
            width: '100%', height: '100%',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 16,
            background: 'transparent',
        }}>
            {/* Pulsing ring loader matching the neon design system */}
            <div style={{ position: 'relative', width: 48, height: 48 }}>
                <div style={{
                    position: 'absolute', inset: 0,
                    borderRadius: '50%',
                    border: '2px solid rgba(0,220,255,0.15)',
                }} />
                <div style={{
                    position: 'absolute', inset: 0,
                    borderRadius: '50%',
                    border: '2px solid transparent',
                    borderTopColor: '#00dcff',
                    animation: 'splineSpinLoader 0.9s linear infinite',
                }} />
                <div style={{
                    position: 'absolute', inset: 8,
                    borderRadius: '50%',
                    border: '1px solid transparent',
                    borderTopColor: 'rgba(0,220,255,0.5)',
                    animation: 'splineSpinLoader 1.4s linear infinite reverse',
                }} />
            </div>
            <span style={{
                fontFamily: "'Orbitron', monospace",
                fontSize: 8, letterSpacing: 4,
                color: 'rgba(0,220,255,0.35)',
            }}>
                LOADING 3D
            </span>

            {/* Keyframes injected inline — avoids CSS module scope issues */}
            <style>{`
        @keyframes splineSpinLoader {
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    )
}

// ── Main export ───────────────────────────────────────────────
export function SplineScene({ scene, style = {}, className = '' }) {
    return (
        <Suspense fallback={<SplineLoader />}>
            <Spline
                scene={scene}
                style={{ width: '100%', height: '100%', ...style }}
                className={className}
            />
        </Suspense>
    )
}
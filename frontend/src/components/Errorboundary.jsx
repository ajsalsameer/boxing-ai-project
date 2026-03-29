import { Component } from 'react'

/**
 * Wraps any component tree. If a child throws during render or in a lifecycle,
 * this catches it and shows a fallback instead of a blank page.
 *
 * Usage:
 *   <ErrorBoundary label="Technique Animator">
 *     <TechniqueAnimator move={move} />
 *   </ErrorBoundary>
 */
export default class ErrorBoundary extends Component {
    constructor(props) {
        super(props)
        this.state = { error: null }
    }

    static getDerivedStateFromError(error) {
        return { error }
    }

    componentDidCatch(error, info) {
        console.error('[ErrorBoundary]', this.props.label || '', error, info)
    }

    render() {
        if (this.state.error) {
            // If a custom fallback was provided (e.g. TechniqueAnimator as Spline fallback), use it
            if (this.props.fallback) return this.props.fallback
            return (
                <div style={{
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    height: '100%', width: '100%',
                    background: '#08080f',
                    border: '1px solid rgba(255,23,68,0.25)',
                    color: 'rgba(255,100,100,0.7)',
                    fontFamily: 'var(--font-hud)',
                    fontSize: '10px', letterSpacing: '2px',
                    gap: '12px', padding: '20px', textAlign: 'center',
                }}>
                    <div style={{ fontSize: '22px' }}>⚠</div>
                    <div>{(this.props.label || 'Component').toUpperCase()} ERROR</div>
                    <div style={{ color: 'rgba(255,255,255,0.2)', fontSize: '9px', maxWidth: 200 }}>
                        {this.state.error?.message || 'Unknown error'}
                    </div>
                    <button
                        onClick={() => this.setState({ error: null })}
                        style={{
                            marginTop: 8, padding: '6px 16px',
                            background: 'transparent',
                            border: '1px solid rgba(255,23,68,0.4)',
                            color: 'rgba(255,100,100,0.7)',
                            fontFamily: 'var(--font-hud)', fontSize: '9px',
                            letterSpacing: '2px', cursor: 'pointer',
                        }}
                    >
                        RETRY
                    </button>
                </div>
            )
        }
        return this.props.children
    }
}
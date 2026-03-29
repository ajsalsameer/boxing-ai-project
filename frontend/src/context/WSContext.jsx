import { createContext, useContext, useRef, useState, useCallback, useEffect } from 'react'

const WSContext = createContext(null)

export function WSProvider({ children }) {
  const wsRef = useRef(null)
  const videoRef = useRef(null)
  const frameTimerRef = useRef(null)
  const reconnectTimer = useRef(null)
  const intentionalClose = useRef(false)

  const [connected, setConnected] = useState(false)
  const [inference, setInference] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [stream, setStream] = useState(null)
  const [currentUser, setCurrentUser] = useState(null)
  const [trainProgress, setTrainProgress] = useState(null)
  const [wsMessages, setWsMessages] = useState([])

  const pushMsg = useCallback((m) => {
    setWsMessages(prev => [...prev.slice(-20), m])
  }, [])

  const stopFrames = useCallback(() => {
    if (frameTimerRef.current) {
      clearTimeout(frameTimerRef.current)
      frameTimerRef.current = null
    }
  }, [])

  // ── WebSocket connect ────────────────────────────────────
  const connect = useCallback(() => {
    const state = wsRef.current?.readyState
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return

    intentionalClose.current = false

    let ws
    try {
      ws = new WebSocket(`ws://${location.host}/ws`)
    } catch {
      reconnectTimer.current = setTimeout(connect, 3000)
      return
    }

    wsRef.current = ws

    ws.onopen = () => {
      if (wsRef.current !== ws) return
      setConnected(true)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
    }

    ws.onmessage = (e) => {
      if (wsRef.current !== ws) return
      let msg
      try { msg = JSON.parse(e.data) } catch { return }

      if (msg.type === 'inference') {
        setInference(msg)
        // train_progress is legacy — stub response from server, always null now
        if (msg.train_progress) {
          setTrainProgress(msg.train_progress)
        } else {
          setTrainProgress(prev => (prev?.training) ? null : prev)
        }
        return
      }

      if (msg.type === 'user_loaded') {
        // v4: server sends markov_label + markov_stats instead of gru_label
        setCurrentUser({
          username: msg.username,
          summary: msg.summary,
          markov_label: msg.markov_label || 'global',
          markov_stats: msg.markov_stats || null,
          // keep gru_label as alias so any old references don't crash
          gru_label: msg.markov_label || 'global',
        })
      }

      if (msg.type === 'record_done') {
        setCurrentUser(prev => prev
          ? {
            ...prev,
            summary: msg.summary || prev.summary,
            markov_stats: msg.markov_stats || prev.markov_stats,
          }
          : prev
        )
      }

      if (msg.type === 'train_started') {
        // GRU training is gone — this is a no-op stub response
        setTrainProgress({ pct: 100, msg: 'Markov active — no training needed.', training: false })
      }
      if (msg.type === 'train_error') setTrainProgress(null)

      pushMsg(msg)
    }

    ws.onclose = () => {
      if (wsRef.current !== ws) return
      setConnected(false)
      stopFrames()
      if (!intentionalClose.current) {
        reconnectTimer.current = setTimeout(connect, 2500)
      }
    }

    ws.onerror = () => { }
  }, [pushMsg, stopFrames])

  const closeWS = useCallback(() => {
    intentionalClose.current = true
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
    const ws = wsRef.current
    wsRef.current = null
    if (ws) ws.close()
    setConnected(false)
  }, [])

  useEffect(() => {
    connect()
    return () => { closeWS(); stopFrames() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Camera ───────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      })
      setStream(s)
      setCameraActive(true)
      if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play() }
      connect()
      return true
    } catch (e) {
      console.error('Camera error:', e)
      return false
    }
  }, [connect])

  const stopCamera = useCallback(() => {
    stopFrames()
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
    setCameraActive(false)
  }, [stream, stopFrames])

  // ── Frame loop ───────────────────────────────────────────
  const canvasRef = useRef(null)
  const getCanvas = useCallback(() => {
    if (!canvasRef.current) canvasRef.current = document.createElement('canvas')
    return canvasRef.current
  }, [])

  const startFrames = useCallback(() => {
    const cvs = getCanvas()
    const ctx = cvs.getContext('2d')
    const loop = () => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      const vid = videoRef.current
      if (!vid || vid.readyState < 2) {
        frameTimerRef.current = setTimeout(loop, 50)
        return
      }
      cvs.width = 320; cvs.height = 240
      ctx.drawImage(vid, 0, 0, 320, 240)
      if (ws.bufferedAmount < 50000) {
        ws.send(JSON.stringify({ type: 'frame', data: cvs.toDataURL('image/jpeg', 0.6) }))
      }
      frameTimerRef.current = setTimeout(loop, 40)   // ~25 fps
    }
    loop()
  }, [getCanvas])

  useEffect(() => {
    if (cameraActive && connected) startFrames()
    else stopFrames()
  }, [cameraActive, connected, startFrames, stopFrames])

  // ── Video ref registration ───────────────────────────────
  const registerVideo = useCallback((el) => {
    videoRef.current = el
    if (el && stream) { el.srcObject = stream; el.play() }
  }, [stream])

  // ── Helpers ──────────────────────────────────────────────
  const send = useCallback((obj) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify(obj))
  }, [])

  const loginUser = useCallback((u) => send({ type: 'set_user', username: u }), [send])
  const startRecording = useCallback(() => send({ type: 'set_mode', mode: 'record' }), [send])
  const stopRecording = useCallback(() => send({ type: 'record_stop' }), [send])

  // trainPersonal is a legacy stub — GRU replaced by Markov (no training needed)
  const trainPersonal = useCallback(() => {
    console.info('trainPersonal: GRU removed, Markov builds automatically from punches.')
  }, [])

  return (
    <WSContext.Provider value={{
      connected, inference, cameraActive, stream,
      currentUser, setCurrentUser,
      trainProgress, wsMessages,
      startCamera, stopCamera, send, registerVideo,
      loginUser, startRecording, stopRecording, trainPersonal,
    }}>
      {children}
    </WSContext.Provider>
  )
}

export const useWS = () => useContext(WSContext)
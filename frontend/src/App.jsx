import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { WSProvider, useWS } from './context/WSContext'
import Navbar   from './components/Navbar'
import Home     from './pages/Home'
import Training from './pages/Training'
import FreePlay from './pages/FreePlay'
import Coach    from './pages/Coach'
import Stats    from './pages/Stats'

function KeyHandler() {
  const { cameraActive, startCamera, stopCamera, send } = useWS()
  useEffect(() => {
    const h = (e) => {
      if (e.target.tagName === 'INPUT') return
      if (e.code === 'Space') { e.preventDefault(); cameraActive ? stopCamera() : startCamera() }
      if (e.code === 'KeyF') send({ type:'set_mode', mode:'freeplay' })
      if (e.code === 'KeyC') send({ type:'set_mode', mode:'coach' })
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [cameraActive, startCamera, stopCamera, send])
  return null
}

function AppInner() {
  return (
    <BrowserRouter>
      <KeyHandler />
      <Navbar />
      <Routes>
        <Route path="/"          element={<Home />}     />
        <Route path="/training"  element={<Training />} />
        <Route path="/freeplay"  element={<FreePlay />} />
        <Route path="/coach"     element={<Coach />}    />
        <Route path="/stats"     element={<Stats />}    />
      </Routes>
    </BrowserRouter>
  )
}

export default function App() {
  return (
    <WSProvider>
      <AppInner />
    </WSProvider>
  )
}
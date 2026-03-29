import { useEffect, useRef } from 'react'
import { useWS } from '../context/WSContext'
import styles from './CameraFeed.module.css'

export default function CameraFeed({ className = '' }) {
  const { inference, cameraActive, registerVideo } = useWS()
  const videoRef = useRef(null)

  useEffect(() => {
    if (videoRef.current) registerVideo(videoRef.current)
  }, [registerVideo])

  const posed = inference?.pose_detected

  return (
    <div className={`${styles.wrap} ${className}`}>
      {/* Hidden video for capture */}
      <video ref={videoRef} autoPlay playsInline muted className={styles.hidden} />

      {/* Processed frame from server (with skeleton drawn) */}
      {inference?.frame
        ? <img src={inference.frame} className={styles.feed} alt="live feed" />
        : <div className={styles.offline}>
            <div className={styles.offlineIcon}>📷</div>
            <div className={styles.offlineText}>
              {cameraActive ? 'INITIALISING…' : 'CAMERA OFFLINE'}
            </div>
          </div>
      }

      {/* Corner brackets */}
      <div className={`${styles.corner} ${styles.tl}`} />
      <div className={`${styles.corner} ${styles.tr}`} />
      <div className={`${styles.corner} ${styles.bl}`} />
      <div className={`${styles.corner} ${styles.br}`} />

      {/* Pose lock indicator */}
      <div className={`${styles.poseBadge} ${posed ? styles.poseOk : styles.poseNo}`}>
        {posed ? '⬡ POSE LOCKED' : '⬡ NO POSE'}
      </div>

      {/* Scan sweep animation */}
      {cameraActive && <div className={styles.scanLine} />}
    </div>
  )
}
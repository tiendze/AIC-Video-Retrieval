import { useEffect, useState } from 'react'
import { getVideo, getVideoKeyframes } from '../api/client'
import type { SearchHit, VideoInfo, VideoKeyframe } from '../api/types'
import styles from './KeyframeDetail.module.css'

export interface KeyframeDetailProps {
  hit: SearchHit
  onClose: () => void
  onFindSimilar: (hit: SearchHit) => void
}

// PRD §F9 (C): full-res image + "Play YouTube" + "Find Similar Images".
// The neighboring-keyframes strip is not in the paper -- added because a
// Textual KIS top-1 hit is often a few shots off from the real answer, and
// it reuses GET /api/videos/{id}/keyframes as-is (plan §P1.6 backend gap
// 1.4). Cut on request; it costs nothing extra to keep for now.
export function KeyframeDetail({ hit, onClose, onFindSimilar }: KeyframeDetailProps) {
  const [video, setVideo] = useState<VideoInfo | null>(null)
  const [neighbors, setNeighbors] = useState<VideoKeyframe[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setVideo(null)
    setNeighbors([])
    setError(null)

    Promise.all([getVideo(hit.video_id), getVideoKeyframes(hit.video_id)])
      .then(([videoInfo, keyframes]) => {
        if (cancelled) return
        setVideo(videoInfo)
        setNeighbors(keyframes.keyframes)
      })
      .catch(() => {
        if (!cancelled) setError('Could not load video details')
      })

    return () => {
      cancelled = true
    }
  }, [hit.video_id])

  const youtubeUrl = video ? `${video.watch_url}&t=${Math.floor(hit.timestamp)}s` : null

  return (
    <div className={styles.overlay} role="dialog" aria-label="Keyframe detail">
      <div className={styles.panel}>
        <button type="button" className={styles.close} onClick={onClose} aria-label="Close">
          ✕
        </button>
        <img className={styles.image} src={hit.image_url} alt={`${hit.video_id} @ ${hit.timestamp}s`} />
        <div className={styles.meta}>
          <div>
            <strong>{hit.video_id}</strong> · frame {hit.frame_idx} · {hit.timestamp}s
          </div>
          {video && <div className={styles.title}>{video.title}</div>}
          {error && <div className={styles.error}>{error}</div>}
          <div className={styles.actions}>
            {youtubeUrl && (
              <a className={styles.playButton} href={youtubeUrl} target="_blank" rel="noreferrer">
                ▶ Play YouTube
              </a>
            )}
            <button
              type="button"
              className={styles.similarButton}
              onClick={() => {
                // This panel is a full-screen overlay (.overlay: position:
                // fixed; inset: 0) -- without closing it, the grid it
                // triggers an update on stays hidden underneath, making the
                // click look like it did nothing.
                onFindSimilar(hit)
                onClose()
              }}
            >
              Find Similar Images
            </button>
          </div>
        </div>
        {neighbors.length > 0 && (
          <div className={styles.neighbors}>
            {neighbors.map((kf) => (
              <img
                key={kf.keyframe_id}
                className={kf.keyframe_id === hit.keyframe_id ? styles.neighborActive : styles.neighbor}
                src={kf.thumbnail_url}
                alt={`frame ${kf.frame_idx}`}
                title={`${kf.timestamp}s`}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

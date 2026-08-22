import { useMemo, useState } from 'react'
import { getVideo, getVideoKeyframes } from '../api/client'
import type { SearchHit, VideoInfo, VideoKeyframe } from '../api/types'
import styles from './FrameChecker.module.css'

export interface FrameCheckerProps {
  onOpenDetail: (hit: SearchHit) => void
}

export function FrameChecker({ onOpenDetail }: FrameCheckerProps) {
  const [videoId, setVideoId] = useState('')
  const [frameIdx, setFrameIdx] = useState('')
  const [neighborCount, setNeighborCount] = useState(3)
  const [video, setVideo] = useState<VideoInfo | null>(null)
  const [frames, setFrames] = useState<VideoKeyframe[]>([])
  const [centerIndex, setCenterIndex] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const requestedFrame = Number(frameIdx)

  const visibleFrames = useMemo(() => {
    if (centerIndex === null) return []
    const start = Math.max(0, centerIndex - neighborCount)
    const end = Math.min(frames.length, centerIndex + neighborCount + 1)
    return frames.slice(start, end).map((frame, index) => ({
      frame,
      absoluteIndex: start + index,
    }))
  }, [centerIndex, frames, neighborCount])

  function runCheck(): void {
    const cleanVideoId = videoId.trim()
    const rawFrame = Number(frameIdx)
    if (!cleanVideoId || !Number.isFinite(rawFrame) || rawFrame < 0) return
    setError(null)
    setVideo(null)
    setFrames([])
    setCenterIndex(null)
    Promise.all([getVideo(cleanVideoId), getVideoKeyframes(cleanVideoId)])
      .then(([videoInfo, keyframeResponse]) => {
        const keyframes = keyframeResponse.keyframes
        setVideo(videoInfo)
        setFrames(keyframes)
        const nearest = nearestFrameIndex(keyframes, rawFrame)
        setCenterIndex(nearest)
      })
      .catch(() => setError('Could not load frames for that video.'))
  }

  return (
    <section className={styles.panel} aria-label="visual frame checker">
      <div className={styles.controls}>
        <label>
          <span>Video ID</span>
          <input
            aria-label="checker video id"
            value={videoId}
            placeholder="L25_V076"
            onChange={(event) => setVideoId(event.currentTarget.value)}
          />
        </label>
        <label>
          <span>Frame</span>
          <input
            aria-label="checker frame"
            type="number"
            min={0}
            value={frameIdx}
            placeholder="18432"
            onChange={(event) => setFrameIdx(event.currentTarget.value)}
          />
        </label>
        <label>
          <span>Neighbors</span>
          <input
            aria-label="checker neighbors"
            type="number"
            min={1}
            max={8}
            value={neighborCount}
            onChange={(event) => setNeighborCount(Number(event.currentTarget.value))}
          />
        </label>
        <button type="button" onClick={runCheck}>
          Check Frame
        </button>
      </div>
      {video && <div className={styles.title}>{video.video_id} · {video.title}</div>}
      {error && <div className={styles.error}>{error}</div>}
      {visibleFrames.length > 0 && centerIndex !== null && (
        <div className={styles.frames}>
          {visibleFrames.map(({ frame, absoluteIndex }) => {
            const isCenter = absoluteIndex === centerIndex
            const delta = frame.frame_idx - requestedFrame
            return (
              <button
                type="button"
                key={frame.keyframe_id}
                className={`${styles.frame} ${isCenter ? styles.center : ''}`}
                onClick={() => onOpenDetail(frameToHit(videoId.trim(), frame))}
              >
                <img src={frame.thumbnail_url} alt={`frame ${frame.frame_idx}`} />
                <span>
                  {isCenter ? 'center' : delta < 0 ? 'prev' : 'next'} · frame {frame.frame_idx}
                </span>
                {isCenter && delta !== 0 && <small>nearest {delta > 0 ? '+' : ''}{delta}</small>}
              </button>
            )
          })}
        </div>
      )}
    </section>
  )
}

function nearestFrameIndex(frames: VideoKeyframe[], targetFrameIdx: number): number | null {
  if (frames.length === 0) return null
  let bestIndex = 0
  let bestDistance = Math.abs(frames[0]!.frame_idx - targetFrameIdx)
  for (let index = 1; index < frames.length; index += 1) {
    const distance = Math.abs(frames[index]!.frame_idx - targetFrameIdx)
    if (distance < bestDistance) {
      bestDistance = distance
      bestIndex = index
    }
  }
  return bestIndex
}

function frameToHit(videoId: string, frame: VideoKeyframe): SearchHit {
  return {
    keyframe_id: frame.keyframe_id,
    video_id: videoId,
    score: 0,
    frame_idx: frame.frame_idx,
    timestamp: frame.timestamp,
    thumbnail_path: '',
    thumbnail_url: frame.thumbnail_url,
    image_url: frame.image_url,
  }
}

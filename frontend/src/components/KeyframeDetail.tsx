import { useEffect, useMemo, useRef, useState } from 'react'
import { getVideo, getVideoKeyframes } from '../api/client'
import type { AsrEvidenceSegment, SearchHit, VideoInfo, VideoKeyframe } from '../api/types'
import styles from './KeyframeDetail.module.css'

export interface KeyframeDetailProps {
  hit: SearchHit
  onClose: () => void
  onFindSimilar: (imageUrl: string) => void
  highlightedFrames?: DetailFrame[]
}

export interface DetailFrame {
  keyframe_id: string
  video_id: string
  frame_idx: number
  timestamp: number
  thumbnail_url: string
  image_url: string
}

function frameFromHit(hit: SearchHit): DetailFrame {
  return {
    keyframe_id: hit.keyframe_id,
    video_id: hit.video_id,
    frame_idx: hit.frame_idx,
    timestamp: hit.timestamp,
    thumbnail_url: hit.thumbnail_url,
    image_url: hit.image_url,
  }
}

function frameFromVideoKeyframe(videoId: string, keyframe: VideoKeyframe): DetailFrame {
  return {
    keyframe_id: keyframe.keyframe_id,
    video_id: videoId,
    frame_idx: keyframe.frame_idx,
    timestamp: keyframe.timestamp,
    thumbnail_url: keyframe.thumbnail_url,
    image_url: keyframe.image_url,
  }
}

// oxlint-disable-next-line react/only-export-components -- tested pure helper required by detail UI
export function nearestKeyframe(
  segment: AsrEvidenceSegment,
  keyframes: VideoKeyframe[],
): VideoKeyframe | null {
  const midpoint = (segment.start_time + segment.end_time) / 2
  return keyframes.reduce<VideoKeyframe | null>((best, frame) => {
    if (!best) return frame
    const frameDistance = Math.abs(frame.timestamp - midpoint)
    const bestDistance = Math.abs(best.timestamp - midpoint)
    return frameDistance < bestDistance ||
      (frameDistance === bestDistance && frame.timestamp < best.timestamp)
      ? frame
      : best
  }, null)
}

// oxlint-disable-next-line react/only-export-components -- tested pure helper required by detail UI
export function isInsideEvidence(
  frame: { timestamp: number },
  segment: AsrEvidenceSegment,
): boolean {
  return frame.timestamp >= segment.start_time && frame.timestamp <= segment.end_time
}

// PRD §F9 (C): full-res image + "Play YouTube" + "Find Similar Images".
// The neighboring-keyframes strip is not in the paper -- added because a
// Textual KIS top-1 hit is often a few shots off from the real answer, and
// it reuses GET /api/videos/{id}/keyframes as-is (plan §P1.6 backend gap
// 1.4). Cut on request; it costs nothing extra to keep for now.
export function KeyframeDetail({
  hit,
  highlightedFrames = [],
  onClose,
  onFindSimilar,
}: KeyframeDetailProps) {
  const [video, setVideo] = useState<VideoInfo | null>(null)
  const [neighbors, setNeighbors] = useState<VideoKeyframe[]>([])
  const [currentFrame, setCurrentFrame] = useState<DetailFrame>(() => frameFromHit(hit))
  const [activeEvidence, setActiveEvidence] = useState<AsrEvidenceSegment | null>(null)
  const [error, setError] = useState<string | null>(null)
  const stripRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    setVideo(null)
    setNeighbors([])
    setError(null)
    setCurrentFrame(frameFromHit(hit))
    setActiveEvidence(null)

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
  }, [hit])

  const stripFrames = useMemo(() => {
    const videoFrames = neighbors.map((keyframe) => frameFromVideoKeyframe(hit.video_id, keyframe))
    const framesById = new Map<string, DetailFrame>()
    for (const frame of [...videoFrames, ...highlightedFrames, currentFrame]) {
      framesById.set(frame.keyframe_id, frame)
    }
    return Array.from(framesById.values()).sort((a, b) => a.timestamp - b.timestamp)
  }, [currentFrame, highlightedFrames, hit.video_id, neighbors])

  const highlightedKeyframeIds = useMemo(
    () => new Set(highlightedFrames.map((frame) => frame.keyframe_id)),
    [highlightedFrames],
  )

  const currentIndex = stripFrames.findIndex(
    (frame) => frame.keyframe_id === currentFrame.keyframe_id,
  )
  const canGoPrev = currentIndex > 0
  const canGoNext = currentIndex >= 0 && currentIndex < stripFrames.length - 1
  const youtubeUrl = video ? `${video.watch_url}&t=${Math.floor(currentFrame.timestamp)}s` : null

  useEffect(() => {
    const active = stripRef.current?.querySelector('[data-active="true"]') as HTMLElement | null
    active?.scrollIntoView?.({ block: 'nearest', inline: 'center' })
  }, [currentFrame.keyframe_id, stripFrames])

  function goToFrame(offset: number): void {
    const next = stripFrames[currentIndex + offset]
    if (next) setCurrentFrame(next)
  }

  function selectEvidence(segment: AsrEvidenceSegment): void {
    setActiveEvidence(segment)
    const nearest = nearestKeyframe(segment, neighbors)
    if (nearest) setCurrentFrame(frameFromVideoKeyframe(hit.video_id, nearest))
  }

  return (
    <div className={styles.overlay} role="dialog" aria-label="Keyframe detail">
      <div className={styles.panel}>
        <button type="button" className={styles.close} onClick={onClose} aria-label="Close">
          ✕
        </button>
        <div className={styles.imageStage}>
          <img
            className={styles.image}
            src={currentFrame.image_url}
            alt={`${currentFrame.video_id} @ ${currentFrame.timestamp}s`}
          />
        </div>
        <div className={styles.meta}>
          <div>
            <strong>{currentFrame.video_id}</strong> · <span>frame {currentFrame.frame_idx}</span> ·{' '}
            {currentFrame.timestamp}s
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
                onFindSimilar(currentFrame.image_url)
                onClose()
              }}
            >
              Find Similar Images
            </button>
          </div>
        </div>
        {hit.asr_evidence && (
          <section className={styles.asrEvidence} aria-label="ASR evidence">
            {hit.asr_evidence.segments.slice(0, 3).map((segment) => (
              <button
                type="button"
                key={segment.segment_id}
                className={styles.asrSegment}
                onClick={() => selectEvidence(segment)}
                aria-pressed={activeEvidence?.segment_id === segment.segment_id}
                aria-label={`${segment.start_time}s–${segment.end_time}s: ${segment.text}`}
              >
                <span>
                  {segment.start_time}s–{segment.end_time}s
                </span>
                <span>{segment.text}</span>
              </button>
            ))}
          </section>
        )}
        {stripFrames.length > 0 && (
          <div className={styles.neighborStrip}>
            <button
              type="button"
              className={styles.navButton}
              onClick={() => goToFrame(-1)}
              disabled={!canGoPrev}
              aria-label="Previous keyframe"
            >
              ‹
            </button>
            <div className={styles.neighbors} ref={stripRef}>
              {stripFrames.map((frame) => {
                const active = frame.keyframe_id === currentFrame.keyframe_id
                const highlighted =
                  highlightedKeyframeIds.has(frame.keyframe_id) ||
                  (activeEvidence !== null && isInsideEvidence(frame, activeEvidence))
                return (
                  <button
                    type="button"
                    key={frame.keyframe_id}
                    className={[
                      styles.neighbor,
                      highlighted ? styles.neighborHighlighted : '',
                      active ? styles.neighborActive : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                    onClick={() => setCurrentFrame(frame)}
                    data-active={active}
                    data-highlighted={highlighted || undefined}
                    aria-current={active ? 'true' : undefined}
                    aria-label={`Show frame ${frame.frame_idx}`}
                    title={`${frame.timestamp}s`}
                  >
                    <img src={frame.thumbnail_url} alt={`frame ${frame.frame_idx}`} />
                  </button>
                )
              })}
            </div>
            <button
              type="button"
              className={styles.navButton}
              onClick={() => goToFrame(1)}
              disabled={!canGoNext}
              aria-label="Next keyframe"
            >
              ›
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

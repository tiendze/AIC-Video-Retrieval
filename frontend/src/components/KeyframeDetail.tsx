import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getVideo, getVideoKeyframes } from '../api/client'
import type { AsrEvidenceSegment, SearchHit, VideoInfo, VideoKeyframe } from '../api/types'
import styles from './KeyframeDetail.module.css'

interface YouTubePlayer {
  getCurrentTime: () => number
  seekTo: (seconds: number, allowSeekAhead: boolean) => void
  cueVideoById?: (options: { videoId: string; startSeconds: number }) => void
  loadVideoById?: (options: { videoId: string; startSeconds: number }) => void
  mute?: () => void
  unMute?: () => void
  pauseVideo?: () => void
  destroy: () => void
}

interface YouTubePlayerEvent {
  target: YouTubePlayer
  data?: number
}

interface YouTubePlayerOptions {
  videoId: string
  playerVars?: Record<string, number | string>
  events?: {
    onReady?: (event: YouTubePlayerEvent) => void
    onStateChange?: (event: YouTubePlayerEvent) => void
  }
}

declare global {
  interface Window {
    YT?: {
      Player: new (element: HTMLElement | string, options: YouTubePlayerOptions) => YouTubePlayer
      PlayerState: {
        PLAYING: number
        PAUSED: number
      }
    }
    onYouTubeIframeAPIReady?: () => void
  }
}

let youtubeIframeApiPromise: Promise<void> | null = null

function loadYouTubeIframeApi(): Promise<void> {
  if (window.YT?.Player) return Promise.resolve()
  if (!youtubeIframeApiPromise) {
    youtubeIframeApiPromise = new Promise((resolve, reject) => {
      const previousReady = window.onYouTubeIframeAPIReady
      window.onYouTubeIframeAPIReady = () => {
        previousReady?.()
        resolve()
      }
      const existingScript = document.querySelector(
        'script[src="https://www.youtube.com/iframe_api"]',
      )
      if (!existingScript) {
        const script = document.createElement('script')
        script.src = 'https://www.youtube.com/iframe_api'
        script.async = true
        script.onerror = () => reject(new Error('Could not load YouTube player'))
        document.head.appendChild(script)
      }
    })
  }
  return youtubeIframeApiPromise
}

export interface KeyframeDetailProps {
  hit: SearchHit
  onClose: () => void
  onFindSimilar?: (imageUrl: string) => void
  highlightedFrames?: DetailFrame[]
  mediaOverride?: KeyframeDetailMedia
}

export type KeyframeDetailMedia =
  | { kind: 'image' }
  | {
      kind: 'youtube'
      youtubeUrl: string
      timestampSeconds: number
      fps: number | null
      title?: string | null
    }
  | { kind: 'unavailable'; message: string }

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

// oxlint-disable-next-line react/only-export-components -- tested pure helper for player routing
export function youtubeVideoIdFromUrl(watchUrl: string): string | null {
  try {
    const url = new URL(watchUrl)
    const host = url.hostname.replace(/^www\./, '')
    if (host === 'youtu.be') return url.pathname.split('/').filter(Boolean)[0] ?? null
    if (host === 'youtube.com' || host.endsWith('.youtube.com')) {
      const watchId = url.searchParams.get('v')
      if (watchId) return watchId
      const embedMatch = url.pathname.match(/^\/embed\/([^/?]+)/)
      if (embedMatch) return embedMatch[1]
      const shortsMatch = url.pathname.match(/^\/shorts\/([^/?]+)/)
      if (shortsMatch) return shortsMatch[1]
    }
  } catch {
    return null
  }
  return null
}

function youtubeUrlAtTimestamp(watchUrl: string, timestamp: number): string {
  const rounded = Math.floor(timestamp)
  try {
    const url = new URL(watchUrl)
    url.searchParams.set('t', `${rounded}s`)
    return url.toString()
  } catch {
    const separator = watchUrl.includes('?') ? '&' : '?'
    return `${watchUrl}${separator}t=${rounded}s`
  }
}

function formatSeconds(seconds: number): string {
  return seconds.toFixed(3).replace(/\.?0+$/, '')
}

// oxlint-disable-next-line react/only-export-components -- tested pure helper for YouTube frame lookup
export function frameIndexFromTimestamp(timestamp: number, fps: number): number | null {
  if (!Number.isFinite(timestamp) || timestamp < 0) return null
  if (!Number.isFinite(fps) || fps <= 0) return null
  return Math.round(timestamp * fps)
}

const DISTANCE_TIE_TOLERANCE_SECONDS = 1e-9

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
    const distanceDelta = frameDistance - bestDistance
    return distanceDelta < -DISTANCE_TIE_TOLERANCE_SECONDS ||
      (Math.abs(distanceDelta) <= DISTANCE_TIE_TOLERANCE_SECONDS &&
        frame.timestamp < best.timestamp)
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
  mediaOverride,
}: KeyframeDetailProps) {
  const media: KeyframeDetailMedia = mediaOverride ?? { kind: 'image' }
  const [video, setVideo] = useState<VideoInfo | null>(null)
  const [neighbors, setNeighbors] = useState<VideoKeyframe[]>([])
  const [currentFrame, setCurrentFrame] = useState<DetailFrame>(() => frameFromHit(hit))
  const [activeEvidence, setActiveEvidence] = useState<AsrEvidenceSegment | null>(null)
  const [playerOpen, setPlayerOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const stripRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setCurrentFrame(frameFromHit(hit))
    setActiveEvidence(null)
    setPlayerOpen(false)
  }, [hit])

  useEffect(() => {
    let cancelled = false
    setVideo(null)
    setNeighbors([])
    setError(null)
    if (media.kind !== 'image') return undefined

    Promise.all([getVideo(currentFrame.video_id), getVideoKeyframes(currentFrame.video_id)])
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
  }, [currentFrame.video_id, media.kind])

  const stripFrames = useMemo(() => {
    const videoFrames = neighbors.map((keyframe) =>
      frameFromVideoKeyframe(currentFrame.video_id, keyframe),
    )
    const framesById = new Map<string, DetailFrame>()
    for (const frame of [...videoFrames, ...highlightedFrames, currentFrame]) {
      framesById.set(frame.keyframe_id, frame)
    }
    return Array.from(framesById.values()).sort((a, b) => a.timestamp - b.timestamp)
  }, [currentFrame, highlightedFrames, neighbors])

  const highlightedKeyframeIds = useMemo(
    () => new Set(highlightedFrames.map((frame) => frame.keyframe_id)),
    [highlightedFrames],
  )

  const currentIndex = stripFrames.findIndex(
    (frame) => frame.keyframe_id === currentFrame.keyframe_id,
  )
  const canGoPrev = currentIndex > 0
  const canGoNext = currentIndex >= 0 && currentIndex < stripFrames.length - 1
  const youtubeUrl =
    video && video.video_id === currentFrame.video_id
      ? youtubeUrlAtTimestamp(video.watch_url, currentFrame.timestamp)
      : null
  const youtubeVideoId =
    video && video.video_id === currentFrame.video_id
      ? youtubeVideoIdFromUrl(video.watch_url)
      : null
  const detailTimestamp =
    media.kind === 'youtube' ? media.timestampSeconds : currentFrame.timestamp

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
          {media.kind === 'youtube' ? (
            <InlineYouTubePreview
              videoId={currentFrame.video_id}
              frameIdx={currentFrame.frame_idx}
              youtubeUrl={media.youtubeUrl}
              timestampSeconds={media.timestampSeconds}
              fps={media.fps}
            />
          ) : media.kind === 'unavailable' ? (
            <div className={styles.unavailableStage}>{media.message}</div>
          ) : (
            <img
              className={styles.image}
              src={currentFrame.image_url}
              alt={`${currentFrame.video_id} @ ${currentFrame.timestamp}s`}
              decoding="async"
            />
          )}
        </div>
        <div className={styles.meta}>
          <div>
            <strong>{currentFrame.video_id}</strong> · <span>frame {currentFrame.frame_idx}</span> ·{' '}
            {formatSeconds(detailTimestamp)}s
          </div>
          {media.kind === 'youtube' && (
            <div className={styles.title}>
              Time: {formatSeconds(media.timestampSeconds)}s
              {media.fps ? ` · FPS: ${formatSeconds(media.fps)}` : ''}
            </div>
          )}
          {video && <div className={styles.title}>{video.title}</div>}
          {error && <div className={styles.error}>{error}</div>}
          <div className={styles.actions}>
            {media.kind === 'image' && youtubeVideoId && (
              <button
                type="button"
                className={styles.playButton}
                onClick={() => setPlayerOpen(true)}
              >
                ▶ Play Video
              </button>
            )}
            {media.kind === 'image' && youtubeUrl && (
              <a className={styles.playButton} href={youtubeUrl} target="_blank" rel="noreferrer">
                Open YouTube
              </a>
            )}
            {media.kind === 'image' && onFindSimilar && (
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
            )}
          </div>
        </div>
        {media.kind === 'image' && hit.asr_evidence && (
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
        {media.kind === 'image' && stripFrames.length > 0 && (
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
                    title={`${formatSeconds(frame.timestamp)}s`}
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
        {media.kind === 'image' && playerOpen && video && youtubeVideoId && (
          <YouTubeFrameIndexModal
            videoId={currentFrame.video_id}
            youtubeVideoId={youtubeVideoId}
            title={video.title}
            fps={video.fps}
            initialTimestamp={currentFrame.timestamp}
            onClose={() => setPlayerOpen(false)}
          />
        )}
      </div>
    </div>
  )
}

interface InlineYouTubePreviewProps {
  videoId: string
  frameIdx: number
  youtubeUrl: string
  timestampSeconds: number
  fps: number | null
}

function InlineYouTubePreview({
  videoId,
  frameIdx,
  youtubeUrl,
  timestampSeconds,
  fps,
}: InlineYouTubePreviewProps) {
  const playerHostRef = useRef<HTMLDivElement>(null)
  const playerRef = useRef<YouTubePlayer | null>(null)
  const youtubeVideoId = youtubeVideoIdFromUrl(youtubeUrl)
  const [playerError, setPlayerError] = useState<string | null>(null)

  useEffect(() => {
    if (!youtubeVideoId) {
      setPlayerError('Youtube Not Available')
      return undefined
    }

    let cancelled = false
    let createdPlayer: YouTubePlayer | null = null
    let pauseTimer: number | null = null
    setPlayerError(null)

    function pauseAtRequestedFrame(player: YouTubePlayer): void {
      player.seekTo(timestampSeconds, true)
      player.pauseVideo?.()
      player.unMute?.()
    }

    function schedulePause(player: YouTubePlayer, delayMs = 420): void {
      if (pauseTimer !== null) window.clearTimeout(pauseTimer)
      pauseTimer = window.setTimeout(() => {
        if (!cancelled) pauseAtRequestedFrame(player)
      }, delayMs)
    }

    loadYouTubeIframeApi()
      .then(() => {
        if (cancelled || !playerHostRef.current || !window.YT?.Player) return
        createdPlayer = new window.YT.Player(playerHostRef.current, {
          videoId: youtubeVideoId,
          playerVars: {
            autoplay: 1,
            controls: 1,
            mute: 1,
            modestbranding: 1,
            playsinline: 1,
            rel: 0,
            start: Math.max(0, Math.floor(timestampSeconds)),
          },
          events: {
            onReady: (event) => {
              playerRef.current = event.target
              event.target.mute?.()
              if (event.target.loadVideoById) {
                event.target.loadVideoById({
                  videoId: youtubeVideoId,
                  startSeconds: timestampSeconds,
                })
                schedulePause(event.target)
              } else {
                event.target.cueVideoById?.({
                  videoId: youtubeVideoId,
                  startSeconds: timestampSeconds,
                })
                pauseAtRequestedFrame(event.target)
              }
            },
            onStateChange: (event) => {
              if (event.data === window.YT?.PlayerState.PLAYING) {
                schedulePause(event.target, 260)
              }
            },
          },
        })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setPlayerError(error instanceof Error ? error.message : 'Youtube Not Available')
        }
      })

    return () => {
      cancelled = true
      if (pauseTimer !== null) window.clearTimeout(pauseTimer)
      const player = playerRef.current ?? createdPlayer
      player?.destroy()
      playerRef.current = null
    }
  }, [timestampSeconds, youtubeVideoId])

  if (!youtubeVideoId) {
    return <div className={styles.unavailableStage}>Youtube Not Available</div>
  }

  return (
    <div className={styles.youtubeStage}>
      <div className={styles.youtubeMeta}>
        <strong>Video: {videoId}</strong>
        <span>Frame: {frameIdx}</span>
        <span>Time: {formatSeconds(timestampSeconds)}s</span>
        {fps && <span>FPS: {formatSeconds(fps)}</span>}
      </div>
      <div className={styles.inlinePlayerFrame} ref={playerHostRef} />
      {playerError && <div className={styles.error}>{playerError}</div>}
    </div>
  )
}

interface FrameIndexResult {
  timestamp: number
  fps: number
  frameIdx: number
}

interface YouTubeFrameIndexModalProps {
  videoId: string
  youtubeVideoId: string
  title: string
  fps: number
  initialTimestamp: number
  onClose: () => void
}

function YouTubeFrameIndexModal({
  videoId,
  youtubeVideoId,
  title,
  fps,
  initialTimestamp,
  onClose,
}: YouTubeFrameIndexModalProps) {
  const playerHostRef = useRef<HTMLDivElement>(null)
  const playerRef = useRef<YouTubePlayer | null>(null)
  const [currentTime, setCurrentTime] = useState(initialTimestamp)
  const [playerReady, setPlayerReady] = useState(false)
  const [playerError, setPlayerError] = useState<string | null>(null)
  const [frameIndexResult, setFrameIndexResult] = useState<FrameIndexResult | null>(null)
  const [frameIndexError, setFrameIndexError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let cancelled = false
    let createdPlayer: YouTubePlayer | null = null
    setPlayerReady(false)
    setPlayerError(null)

    loadYouTubeIframeApi()
      .then(() => {
        if (cancelled || !playerHostRef.current || !window.YT?.Player) return
        createdPlayer = new window.YT.Player(playerHostRef.current, {
          videoId: youtubeVideoId,
          playerVars: {
            autoplay: 1,
            modestbranding: 1,
            rel: 0,
            start: Math.max(0, Math.floor(initialTimestamp)),
          },
          events: {
            onReady: (event) => {
              playerRef.current = event.target
              event.target.seekTo(initialTimestamp, true)
              setCurrentTime(event.target.getCurrentTime())
              setPlayerReady(true)
            },
            onStateChange: (event) => {
              if (event.data === window.YT?.PlayerState.PAUSED) {
                setCurrentTime(event.target.getCurrentTime())
              }
            },
          },
        })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setPlayerError(error instanceof Error ? error.message : 'Could not load YouTube player')
        }
      })

    return () => {
      cancelled = true
      const player = playerRef.current ?? createdPlayer
      player?.destroy()
      playerRef.current = null
    }
  }, [initialTimestamp, youtubeVideoId])

  const handleGetFrameIndex = useCallback((): void => {
    if (!playerReady) return
    const timestamp = playerRef.current?.getCurrentTime() ?? currentTime
    setCurrentTime(timestamp)
    setCopied(false)
    const frameIdx = frameIndexFromTimestamp(timestamp, fps)
    if (frameIdx === null) {
      setFrameIndexResult(null)
      setFrameIndexError(`FPS metadata not found for ${videoId}`)
      return
    }
    setFrameIndexError(null)
    setFrameIndexResult({ timestamp, fps, frameIdx })
  }, [currentTime, fps, playerReady, videoId])

  async function copyFrameIndex(): Promise<void> {
    if (!frameIndexResult) return
    await navigator.clipboard?.writeText(String(frameIndexResult.frameIdx))
    setCopied(true)
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent): void {
      const target = event.target as HTMLElement | null
      const tagName = target?.tagName
      const isTyping =
        tagName === 'INPUT' || tagName === 'TEXTAREA' || target?.isContentEditable === true
      if (isTyping || event.key.toLowerCase() !== 'c') return
      event.preventDefault()
      handleGetFrameIndex()
    }

    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [handleGetFrameIndex])

  return (
    <div className={styles.playerBackdrop} role="dialog" aria-label="Video player">
      <div className={styles.playerPanel}>
        <div className={styles.playerHeader}>
          <div>
            <strong>{videoId}</strong>
            <span>{title}</span>
          </div>
          <button
            type="button"
            className={styles.playerClose}
            onClick={onClose}
            aria-label="Close video player"
          >
            ✕
          </button>
        </div>
        <div className={styles.playerFrame} ref={playerHostRef} />
        <div className={styles.playerStatus}>Current: {formatSeconds(currentTime)} s</div>
        {playerError && <div className={styles.error}>{playerError}</div>}
        {frameIndexError && <div className={styles.error}>{frameIndexError}</div>}
        {frameIndexResult && (
          <div className={styles.frameIndexResult} aria-label="Frame index result">
            <span>Current: {formatSeconds(frameIndexResult.timestamp)} s</span>
            <span>FPS: {formatSeconds(frameIndexResult.fps)}</span>
            <strong>Frame Index: {frameIndexResult.frameIdx}</strong>
            <button
              type="button"
              className={styles.copyButton}
              onClick={() => void copyFrameIndex()}
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        )}
        <div className={styles.captureActions}>
          <button
            type="button"
            className={styles.captureButton}
            onClick={handleGetFrameIndex}
            disabled={!playerReady}
          >
            Get Frame Index
          </button>
        </div>
      </div>
    </div>
  )
}

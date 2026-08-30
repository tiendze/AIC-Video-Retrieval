export function formatTrakeTimestamp(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const remainingSeconds = safeSeconds - hours * 3600 - minutes * 60
  const secondsText = remainingSeconds.toFixed(1).padStart(4, '0')
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secondsText}`
  }
  return `${String(minutes).padStart(2, '0')}:${secondsText}`
}

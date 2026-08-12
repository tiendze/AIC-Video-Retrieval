import { useEffect } from 'react'

// plan §P1.6: "Enter search · / focus ô nhập · ←/→ prev/next · 1/2/3 đổi
// mode · Esc đóng detail. Không kích hoạt khi con trỏ đang trong ô nhập
// (trừ Enter/Esc)."
export interface KeyboardShortcutHandlers {
  onSubmit?: () => void
  onFocusQuery?: () => void
  onNext?: () => void
  onPrev?: () => void
  onSetMode?: (modeIndex: number) => void
  onClose?: () => void
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || target.isContentEditable
}

const MODE_KEYS = new Set(['1', '2', '3'])

export function useKeyboardShortcuts(handlers: KeyboardShortcutHandlers): void {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent): void {
      const typing = isTypingTarget(event.target)

      // Enter/Escape fire even while typing -- every other shortcut is
      // suppressed so a query like "1 2 3 phones" doesn't trigger mode
      // switches or navigation while the user is still composing it.
      if (event.key === 'Escape') {
        handlers.onClose?.()
        return
      }
      if (event.key === 'Enter') {
        if (typing) handlers.onSubmit?.()
        return
      }
      if (typing) return

      if (event.key === '/') {
        event.preventDefault()
        handlers.onFocusQuery?.()
        return
      }
      if (event.key === 'ArrowRight') {
        handlers.onNext?.()
        return
      }
      if (event.key === 'ArrowLeft') {
        handlers.onPrev?.()
        return
      }
      if (MODE_KEYS.has(event.key)) {
        handlers.onSetMode?.(Number(event.key) - 1)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })
}

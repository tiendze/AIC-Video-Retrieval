import type { AppMode } from '../api/types'
import styles from './AppSidebar.module.css'

interface AppSidebarProps {
  mode: AppMode
  collapsed: boolean
  onChange: (mode: AppMode) => void
  onToggleCollapsed: () => void
}

type SidebarIconName = 'pen' | 'chat' | 'scope'

const NAV_ITEMS: {
  mode: AppMode
  label: string
  icon: SidebarIconName
  ariaLabel: string
}[] = [
  { mode: 'kis', label: 'KIS', icon: 'pen', ariaLabel: 'Open KIS workspace' },
  { mode: 'dante', label: 'TRAKE', icon: 'chat', ariaLabel: 'Open TRAKE workspace' },
  {
    mode: 'view-file',
    label: 'View File',
    icon: 'scope',
    ariaLabel: 'Open View File workspace',
  },
]

export function AppSidebar({
  mode,
  collapsed,
  onChange,
  onToggleCollapsed,
}: AppSidebarProps) {
  return (
    <aside
      className={`${styles.sidebar} ${collapsed ? styles.collapsed : ''}`}
      aria-label="Workspace navigation"
    >
      <div className={styles.brand}>
        <span className={styles.brandMark} aria-hidden="true">
          <LogoIcon />
        </span>
        <span className={styles.brandText}>Ctrl Alt Detect</span>
      </div>
      <nav className={styles.nav} aria-label="Workspaces">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.mode}
            type="button"
            aria-current={mode === item.mode ? 'page' : undefined}
            aria-label={item.ariaLabel}
            className={`${styles.navButton} ${mode === item.mode ? styles.active : ''}`}
            data-tooltip={item.label}
            onClick={() => onChange(item.mode)}
          >
            <span className={styles.iconBadge} aria-hidden="true">
              <SidebarIcon name={item.icon} />
            </span>
            <span className={styles.navText}>{item.label}</span>
          </button>
        ))}
      </nav>
      <button
        type="button"
        className={styles.collapseButton}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        data-tooltip={collapsed ? 'Expand' : 'Collapse'}
        onClick={onToggleCollapsed}
      >
        <span aria-hidden="true">{collapsed ? '>' : '<'}</span>
        <span className={styles.navText}>{collapsed ? 'Expand' : 'Collapse'}</span>
      </button>
    </aside>
  )
}

function SidebarIcon({ name }: { name: SidebarIconName }) {
  if (name === 'pen') {
    return (
      <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z" />
      </svg>
    )
  }
  if (name === 'chat') {
    return (
      <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
        <path d="M21 12a7.6 7.6 0 0 1-8 7.5 9.3 9.3 0 0 1-4.4-1.1L3 20l1.7-4.5A7 7 0 0 1 3 12a7.7 7.7 0 0 1 8-7.5h2a7.7 7.7 0 0 1 8 7.5Z" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
      <path d="M12 3v4" />
      <path d="M12 17v4" />
      <path d="M3 12h4" />
      <path d="M17 12h4" />
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1.5" />
    </svg>
  )
}

function LogoIcon() {
  return (
    <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
      <path d="M12 3.5a8.5 8.5 0 1 0 8.5 8.5" />
      <path d="M12 7a5 5 0 1 0 5 5" />
      <path d="M12 10.5a1.5 1.5 0 1 0 1.5 1.5" />
      <path d="M14.5 4.2 21 3l-1.2 6.5" />
    </svg>
  )
}

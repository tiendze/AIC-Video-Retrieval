import { forwardRef, type InputHTMLAttributes } from 'react'
import styles from './SearchBar.module.css'

export interface QueryInputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'className' | 'onChange'> {
  className?: string
  onValueChange: (value: string) => void
}

export const QueryInput = forwardRef<HTMLInputElement, QueryInputProps>(function QueryInput(
  { className, onValueChange, ...props },
  ref,
) {
  return (
    <input
      {...props}
      ref={ref}
      className={[styles.input, className].filter(Boolean).join(' ')}
      onChange={(event) => onValueChange(event.target.value)}
    />
  )
})

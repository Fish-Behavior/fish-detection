import type { ReactNode } from 'react'

const paths = {
  fish: 'M3 12c4-6 11-6 15 0-4 6-11 6-15 0Zm15 0 4-5v10l-4-5ZM8 10h.01',
  dashboard: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  review: 'M9 4H5v16h14V4h-4M9 3h6v4H9zM8 12l2 2 5-5M8 17h7',
  play: 'm8 4 12 8-12 8V4Z',
  pause: 'M8 4v16M16 4v16',
  chat: 'M4 4h16v12H9l-5 4V4ZM8 8h8M8 12h5',
  arrow: 'M4 12h16m-6-6 6 6-6 6',
  upload: 'M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6',
  check: 'm5 12 4 4L19 6',
  reset: 'M4 10a8 8 0 1 1 2 8M4 4v6h6',
  close: 'm6 6 12 12M6 18 18 6',
  info: 'M12 11v6M12 7h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
} as const
export function Icon({
  name,
  size = 18,
}: {
  name: keyof typeof paths
  size?: number
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  )
}
export function Card({
  title,
  eyebrow,
  accessory,
  children,
  className = '',
}: {
  title: string
  eyebrow?: string
  accessory?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`card ${className}`}>
      <div className="card-heading">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h2>{title}</h2>
        </div>
        {accessory}
      </div>
      {children}
    </section>
  )
}

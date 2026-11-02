import type { ReactNode } from 'react'

// The subset of Markdown the research chat writes: paragraphs, **bold**, *italic*, `code`, bullet or numbered lists and
// pipe tables. Everything becomes React text nodes, so model output can never inject HTML.

const inline = (text: string): ReactNode[] =>
  text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) return <em key={i}>{part.slice(1, -1)}</em>
    return part
  })

const cells = (row: string): string[] => row.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
const isRule = (row: string): boolean => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(row)
const bullet = /^\s*(?:[-*•]|\d+[.)])\s+/

export default function ChatMarkdown({ text }: { text: string }) {
  const blocks: ReactNode[] = []
  const lines = text.replace(/\r/g, '').split('\n')
  for (let i = 0; i < lines.length; ) {
    const line = lines[i]
    if (!line.trim()) {
      i++
    } else if (line.includes('|') && i + 1 < lines.length && isRule(lines[i + 1])) {
      const head = cells(line)
      const rows: string[][] = []
      for (i += 2; i < lines.length && lines[i].includes('|') && lines[i].trim(); i++) rows.push(cells(lines[i]))
      blocks.push(
        <div className="chat-table" key={blocks.length}>
          <table>
            <thead><tr>{head.map((c, j) => <th key={j}>{inline(c)}</th>)}</tr></thead>
            <tbody>{rows.map((r, k) => <tr key={k}>{r.map((c, j) => <td key={j}>{inline(c)}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      )
    } else if (bullet.test(line)) {
      const items: string[] = []
      const ordered = /^\s*\d/.test(line)
      for (; i < lines.length && bullet.test(lines[i]); i++) items.push(lines[i].replace(bullet, ''))
      const List = ordered ? 'ol' : 'ul'
      blocks.push(<List key={blocks.length}>{items.map((item, j) => <li key={j}>{inline(item)}</li>)}</List>)
    } else {
      const para: string[] = []
      for (; i < lines.length && lines[i].trim() && !bullet.test(lines[i]) && !(lines[i].includes('|') && isRule(lines[i + 1] ?? '')); i++)
        para.push(lines[i].replace(/^#+\s*/, ''))
      blocks.push(<p key={blocks.length}>{inline(para.join(' '))}</p>)
    }
  }
  return <>{blocks}</>
}

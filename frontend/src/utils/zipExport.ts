export interface ZipTextEntry {
  name: string
  content: string
}

const CRC32_TABLE = new Uint32Array(256)

for (let index = 0; index < CRC32_TABLE.length; index += 1) {
  let value = index
  for (let bit = 0; bit < 8; bit += 1) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1
  }
  CRC32_TABLE[index] = value >>> 0
}

export function uniqueZipEntryNames(names: string[]): string[] {
  const seen = new Map<string, number>()
  return names.map((name) => {
    const count = seen.get(name.toLowerCase()) ?? 0
    seen.set(name.toLowerCase(), count + 1)
    if (count === 0) return name
    const dotIndex = name.lastIndexOf('.')
    if (dotIndex <= 0) return `${name}-${count + 1}`
    return `${name.slice(0, dotIndex)}-${count + 1}${name.slice(dotIndex)}`
  })
}

export function createZipBlob(entries: ZipTextEntry[]): Blob {
  const encoder = new TextEncoder()
  const now = new Date()
  const localParts: Uint8Array[] = []
  const centralParts: Uint8Array[] = []
  let offset = 0

  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.name)
    const contentBytes = encoder.encode(entry.content)
    const crc = crc32(contentBytes)
    const { dosDate, dosTime } = dateToDos(now)
    const localHeader = new Uint8Array(30 + nameBytes.length)
    const localView = new DataView(localHeader.buffer)
    localView.setUint32(0, 0x04034b50, true)
    localView.setUint16(4, 20, true)
    localView.setUint16(6, 0x0800, true)
    localView.setUint16(8, 0, true)
    localView.setUint16(10, dosTime, true)
    localView.setUint16(12, dosDate, true)
    localView.setUint32(14, crc, true)
    localView.setUint32(18, contentBytes.length, true)
    localView.setUint32(22, contentBytes.length, true)
    localView.setUint16(26, nameBytes.length, true)
    localHeader.set(nameBytes, 30)
    localParts.push(localHeader, contentBytes)

    const centralHeader = new Uint8Array(46 + nameBytes.length)
    const centralView = new DataView(centralHeader.buffer)
    centralView.setUint32(0, 0x02014b50, true)
    centralView.setUint16(4, 20, true)
    centralView.setUint16(6, 20, true)
    centralView.setUint16(8, 0x0800, true)
    centralView.setUint16(10, 0, true)
    centralView.setUint16(12, dosTime, true)
    centralView.setUint16(14, dosDate, true)
    centralView.setUint32(16, crc, true)
    centralView.setUint32(20, contentBytes.length, true)
    centralView.setUint32(24, contentBytes.length, true)
    centralView.setUint16(28, nameBytes.length, true)
    centralView.setUint32(42, offset, true)
    centralHeader.set(nameBytes, 46)
    centralParts.push(centralHeader)
    offset += localHeader.length + contentBytes.length
  }

  const centralOffset = offset
  const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0)
  const end = new Uint8Array(22)
  const endView = new DataView(end.buffer)
  endView.setUint32(0, 0x06054b50, true)
  endView.setUint16(8, entries.length, true)
  endView.setUint16(10, entries.length, true)
  endView.setUint32(12, centralSize, true)
  endView.setUint32(16, centralOffset, true)

  const parts = [...localParts, ...centralParts, end].map(toBlobPart)
  return new Blob(parts, { type: 'application/zip' })
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc = CRC32_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

function toBlobPart(bytes: Uint8Array): BlobPart {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
}

function dateToDos(date: Date): { dosDate: number; dosTime: number } {
  const year = Math.max(1980, date.getFullYear())
  return {
    dosDate: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
    dosTime: (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1),
  }
}

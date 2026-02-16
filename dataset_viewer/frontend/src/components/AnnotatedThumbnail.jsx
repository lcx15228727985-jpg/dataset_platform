import { useEffect, useRef, useState } from 'react'
import { getImageUrl, getWorkstationData } from '../api/client'

/** 在缩略图上叠加已保存的椭圆标注 */
export default function AnnotatedThumbnail({
  imageId,
  annotated,
  filename,
  thumb = true,
  onClick,
  className,
  children,
}) {
  const wrapperRef = useRef(null)
  const canvasRef = useRef(null)
  const [boxes, setBoxes] = useState([])

  useEffect(() => {
    if (!annotated || !imageId) return
    getWorkstationData(imageId)
      .then((data) => setBoxes(data.boxes || []))
      .catch(() => {})
  }, [imageId, annotated])

  useEffect(() => {
    if (!boxes.length || !wrapperRef.current || !canvasRef.current) return
    const draw = () => {
      const el = wrapperRef.current
      const canvas = canvasRef.current
      if (!el || !canvas) return
      const rect = el.getBoundingClientRect()
      const w = rect.width
      const h = rect.height
      if (w <= 0 || h <= 0) return
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, w, h)
      boxes.forEach((box) => {
        const nx = box.x ?? 0
        const ny = box.y ?? 0
        const nw = box.w ?? 0.1
        const nh = box.h ?? 0.1
        const isRect = box.type === 'rect'
        ctx.strokeStyle = '#00ff00'
        ctx.lineWidth = 2
        if (isRect) {
          const x = nx * w
          const y = ny * h
          const rw = Math.max(4, nw * w)
          const rh = Math.max(4, nh * h)
          ctx.strokeRect(x, y, rw, rh)
        } else {
          const cx = (nx + nw / 2) * w
          const cy = (ny + nh / 2) * h
          const rx = Math.max(2, (nw / 2) * w)
          const ry = Math.max(2, (nh / 2) * h)
          ctx.beginPath()
          ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI)
          ctx.stroke()
        }
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(wrapperRef.current)
    return () => ro.disconnect()
  }, [boxes])

  return (
    <div
      ref={wrapperRef}
      className={className}
      style={{ position: 'relative', display: 'block' }}
    >
      <img
        src={getImageUrl(imageId, thumb)}
        alt={filename}
        loading="lazy"
        onClick={onClick}
      />
      {annotated && (
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        />
      )}
      {children}
    </div>
  )
}

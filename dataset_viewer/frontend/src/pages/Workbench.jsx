import React, { useEffect, useRef, useCallback, useState } from 'react'
import { useNavigate, useLocation, useSearchParams } from 'react-router-dom'
import { Stage, Layer, Image as KonvaImage, Rect, Ellipse, Transformer } from 'react-konva'
import useImage from 'use-image'
import { useWorkbenchStore } from '../store/workbench'
import {
  getImageUrl,
  getWorkstationData,
  saveWorkstationBoxes,
  clearWorkstationAnnotation,
  getEpisodes,
  getEpisodeImages,
} from '../api/client'
import AnnotatedThumbnail from '../components/AnnotatedThumbnail'
import styles from './Workbench.module.css'


/** 坐标归一化：画布像素 -> 0~1 比例（用于存储） */
function toNormalized(x, y, w, h, canvasW, canvasH) {
  if (!canvasW || !canvasH) return { x: 0, y: 0, w: 0.1, h: 0.1 }
  return {
    x: x / canvasW,
    y: y / canvasH,
    w: Math.max(0.01, w / canvasW),
    h: Math.max(0.01, h / canvasH),
  }
}
/** 坐标反归一化：0~1 -> 画布像素（用于渲染） */
function toPixels(x, y, w, h, canvasW, canvasH) {
  if (!canvasW || !canvasH) return { x: 0, y: 0, w: 50, h: 50 }
  return {
    x: x * canvasW,
    y: y * canvasH,
    w: Math.max(5, w * canvasW),
    h: Math.max(5, h * canvasH),
  }
}

/** 椭圆标注（x,y,w,h 为外接矩形，内部转为 center + radiusX/radiusY） */
function toEllipseParams(px) {
  const cx = px.x + px.w / 2
  const cy = px.y + px.h / 2
  const rx = Math.max(5, px.w / 2)
  const ry = Math.max(5, px.h / 2)
  return { cx, cy, rx, ry }
}
function fromEllipseToBox(cx, cy, rx, ry, canvasW, canvasH) {
  const x = cx - rx
  const y = cy - ry
  const w = rx * 2
  const h = ry * 2
  return toNormalized(x, y, w, h, canvasW, canvasH)
}

/** 单个标注框（椭圆或矩形，可选中、拖拽、缩放） */
function AnnotationBox({ box, canvasW, canvasH, isSelected, onSelect, onChange, listening = true }) {
  const shapeRef = useRef()
  const trRef = useRef()
  const isRect = box.type === 'rect'
  const px = toPixels(box.x ?? 0, box.y ?? 0, box.w ?? 0.1, box.h ?? 0.1, canvasW, canvasH)
  const { cx, cy, rx, ry } = toEllipseParams(px)

  useEffect(() => {
    if (isSelected && trRef.current && shapeRef.current) {
      trRef.current.nodes([shapeRef.current])
      trRef.current.getLayer()?.batchDraw()
    }
  }, [isSelected])

  const handleEllipseDragEnd = useCallback((e) => {
    const node = e.target
    const norm = fromEllipseToBox(node.x(), node.y(), node.radiusX(), node.radiusY(), canvasW, canvasH)
    onChange({ ...box, ...norm })
  }, [box, canvasW, canvasH, onChange])

  const handleEllipseTransformEnd = useCallback((e) => {
    const node = shapeRef.current
    if (!node) return
    const scaleX = node.scaleX()
    const scaleY = node.scaleY()
    node.scaleX(1)
    node.scaleY(1)
    const norm = fromEllipseToBox(node.x(), node.y(), node.radiusX() * scaleX, node.radiusY() * scaleY, canvasW, canvasH)
    onChange({ ...box, ...norm })
  }, [box, canvasW, canvasH, onChange])

  const handleRectDragEnd = useCallback((e) => {
    const node = e.target
    const norm = toNormalized(node.x(), node.y(), node.width(), node.height(), canvasW, canvasH)
    onChange({ ...box, ...norm })
  }, [box, canvasW, canvasH, onChange])

  const handleRectTransformEnd = useCallback((e) => {
    const node = shapeRef.current
    if (!node) return
    const scaleX = node.scaleX()
    const scaleY = node.scaleY()
    node.scaleX(1)
    node.scaleY(1)
    const norm = toNormalized(node.x(), node.y(), node.width() * scaleX, node.height() * scaleY, canvasW, canvasH)
    onChange({ ...box, ...norm })
  }, [box, canvasW, canvasH, onChange])

  const commonProps = { stroke: '#00FF00', strokeWidth: 2, fill: 'rgba(0,255,0,0.15)', draggable: listening, listening, onClick: onSelect, onTap: onSelect }

  return (
    <>
      {isRect ? (
        <Rect
          ref={shapeRef}
          x={px.x}
          y={px.y}
          width={px.w}
          height={px.h}
          {...commonProps}
          onDragEnd={handleRectDragEnd}
          onTransformEnd={handleRectTransformEnd}
        />
      ) : (
        <Ellipse
          ref={shapeRef}
          x={cx}
          y={cy}
          radiusX={rx}
          radiusY={ry}
          {...commonProps}
          onDragEnd={handleEllipseDragEnd}
          onTransformEnd={handleEllipseTransformEnd}
        />
      )}
      {isSelected && <Transformer ref={trRef} />}
    </>
  )
}

/** 只读预览（椭圆或矩形） */
function PreviewBox({ box, canvasW, canvasH }) {
  const isRect = box.type === 'rect'
  const px = toPixels(box.x ?? 0, box.y ?? 0, box.w ?? 0.1, box.h ?? 0.1, canvasW, canvasH)
  const { cx, cy, rx, ry } = toEllipseParams(px)
  const previewProps = { stroke: 'red', strokeWidth: 3, listening: false }
  return isRect ? (
    <Rect x={px.x} y={px.y} width={px.w} height={px.h} {...previewProps} />
  ) : (
    <Ellipse x={cx} y={cy} radiusX={rx} radiusY={ry} {...previewProps} />
  )
}

/** 带 useImage 的 Konva 图片 */
function KonvaImageWithLoader({ src, width, height }) {
  const [img, status] = useImage(src)
  if (status === 'failed') return null
  return <KonvaImage image={img} width={width} height={height} listening={false} />
}

export default function Workbench() {
  const navigate = useNavigate()
  const { state } = useLocation()
  const [searchParams] = useSearchParams()
  const pathFromUrl = searchParams.get('path')
  const runFromUrl = searchParams.get('run') || ''
  const indexFromUrl = parseInt(searchParams.get('index') || '0', 10)

  const { pathId, meta, boxes, loadImage, updateBox, deleteBox, addBox, clearBoxes, clear } = useWorkbenchStore()
  const [selectedId, setSelectedId] = useState(null)
  const [images, setImages] = useState([])
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [saving, setSaving] = useState(false)
  const [addDropdownOpen, setAddDropdownOpen] = useState(false)
  const [drawMode, setDrawMode] = useState(null)
  const [drawStart, setDrawStart] = useState(null)
  const [drawCurrent, setDrawCurrent] = useState(null)
  const addDropdownRef = useRef(null)
  const stageRef = useRef(null)
  const drawStartRef = useRef(null)
  const mountedRef = useRef(true)

  const imgW = meta?.width || 0
  const imgH = meta?.height || 0
  const scale = imgW > 0 && imgH > 0 ? Math.min(1, 800 / Math.max(imgW, imgH)) : 1
  const dispW = Math.round(imgW * scale) || 800
  const dispH = Math.round(imgH * scale) || 600

  const imageUrl = pathId ? getImageUrl(pathId) : null

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false; clear() }
  }, [])

  useEffect(() => {
    if (!pathFromUrl && !state?.imageId) {
      navigate('/', { replace: true })
      return
    }
    const path = state?.imageId || pathFromUrl
    const imgs = state?.images
    const idx = state?.index ?? indexFromUrl
    if (!path) return
    setLoading(true)
    setError(null)
    let imgsResolved = imgs && Array.isArray(imgs) && imgs.length > 0
    if (imgsResolved) {
      setImages(imgs)
      setIndex(idx >= 0 && idx < imgs.length ? idx : 0)
    }
    const fetchWorkstation = () => getWorkstationData(path).then((data) => {
      if (!mountedRef.current) return
      loadImage({
        path_id: data.path_id,
        imageUrl: getImageUrl(data.path_id),
        meta: data.meta,
        boxes: (data.boxes || []).map((b) => ({
          ...b,
          id: b.id || `box_${Date.now()}_${Math.random().toString(36).slice(2)}`,
        })),
      })
    })
    if (imgsResolved) {
      fetchWorkstation().catch((e) => {
        if (mountedRef.current) setError(e.message)
      }).finally(() => {
        if (mountedRef.current) setLoading(false)
      })
      return
    }
    const epMatch = path.split('/').find((p) => /^episode_\d+$/.test(p))
    const epName = epMatch ? 'ep' + epMatch.replace('episode_', '') : null
    const run = runFromUrl || 'unknown'
    getEpisodes(run)
      .then((d) => {
        if (!mountedRef.current) return
        const eps = d.episodes || []
        for (const ep of eps) {
          const list = ep.images || []
          const i = list.findIndex((x) => x.id === path)
          if (i >= 0) {
            setImages(list)
            setIndex(i)
            return getWorkstationData(path)
          }
        }
        if (epName && d.cursorAvailable) {
          return getEpisodeImages(run, epName, null, 80).then(({ items }) => {
            if (!mountedRef.current) return Promise.reject(new Error('unmounted'))
            const i = items.findIndex((x) => x.id === path)
            setImages(items)
            setIndex(i >= 0 ? i : indexFromUrl)
            return getWorkstationData(path)
          })
        }
        return getWorkstationData(path)
      })
      .then((data) => {
        if (!mountedRef.current || !data) return
        loadImage({
          path_id: data.path_id,
          imageUrl: getImageUrl(data.path_id),
          meta: data.meta,
          boxes: (data.boxes || []).map((b) => ({
            ...b,
            id: b.id || `box_${Date.now()}_${Math.random().toString(36).slice(2)}`,
          })),
        })
      })
      .catch((e) => {
        if (mountedRef.current) setError(e.message || '加载失败')
      })
      .finally(() => {
        if (mountedRef.current) setLoading(false)
      })
  }, [pathFromUrl, runFromUrl, indexFromUrl, state?.imageId, state?.images, state?.index, navigate, loadImage, clear])

  const syncImagesAndNavigate = useCallback((nextImages, nextId, nextIdx) => {
    setImages(nextImages)
    saveWorkstationBoxes(pathId, boxes).catch(() => {}).finally(() => {
      navigate(`/annotate?path=${encodeURIComponent(nextId)}&run=${encodeURIComponent(runFromUrl)}&index=${nextIdx}`, {
        state: { imageId: nextId, images: nextImages, index: nextIdx, run: runFromUrl },
        replace: true,
      })
    })
  }, [pathId, boxes, runFromUrl, navigate])

  const goPrev = useCallback(() => {
    if (index <= 0 || !images[index - 1]) return
    const next = images[index - 1]
    const updated = images.map((img, i) => (i === index ? { ...img, annotated: boxes.length > 0 } : img))
    syncImagesAndNavigate(updated, next.id, index - 1)
  }, [index, images, pathId, boxes, runFromUrl, navigate, syncImagesAndNavigate])

  const goNext = useCallback(() => {
    if (index >= images.length - 1 || !images[index + 1]) return
    const next = images[index + 1]
    const updated = images.map((img, i) => (i === index ? { ...img, annotated: boxes.length > 0 } : img))
    syncImagesAndNavigate(updated, next.id, index + 1)
  }, [index, images, pathId, boxes, runFromUrl, navigate, syncImagesAndNavigate])

  const handleSave = useCallback(async () => {
    if (!pathId) return
    setSaving(true)
    try {
      await saveWorkstationBoxes(pathId, boxes)
      const annotated = boxes.length > 0
      setImages((prev) =>
        prev.map((img, i) =>
          i === index ? { ...img, annotated } : img
        )
      )
    } catch (e) {
      alert(e.message)
    } finally {
      setSaving(false)
    }
  }, [pathId, boxes, index])

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (addDropdownRef.current && !addDropdownRef.current.contains(e.target)) setAddDropdownOpen(false)
    }
    if (addDropdownOpen) {
      document.addEventListener('click', handleClickOutside)
      return () => document.removeEventListener('click', handleClickOutside)
    }
  }, [addDropdownOpen])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape' && drawMode) {
        drawStartRef.current = null
        setDrawMode(null)
        setDrawStart(null)
        setDrawCurrent(null)
      } else if (e.key === 'Delete' && selectedId) {
        deleteBox(selectedId)
        setSelectedId(null)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [drawMode, selectedId, deleteBox])

  if (error) {
    return (
      <div className={styles.page}>
        <header className={styles.header}>
          <button type="button" onClick={() => navigate('/')}>← 返回图库</button>
        </header>
        <main className={styles.main}><p className={styles.error}>{error}</p></main>
      </div>
    )
  }
  if (loading || !imageUrl) {
    return (
      <div className={styles.page}>
        <header className={styles.header}>
          <button type="button" onClick={() => navigate('/')}>← 返回图库</button>
        </header>
        <main className={styles.main}><p className={styles.loading}>加载中…</p></main>
      </div>
    )
  }

  const nTotal = images.length
  const annotatedCount = images.filter((i) => i.annotated).length
  const currentFilename = images[index]?.filename || pathId?.split('/').pop() || ''

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <button type="button" onClick={() => navigate('/')}>← 返回图库</button>
        <span className={styles.title}>标注工作台</span>
        <span className={styles.meta}>
          {runFromUrl} · 第 {index + 1}/{nTotal} 张 · 已标注 {annotatedCount}/{nTotal} · {currentFilename}
        </span>
      </header>
      <div className={styles.thumbnailStrip}>
        <span className={styles.stripLabel}>本 ep 预览：</span>
        <div className={styles.stripScroll}>
          {images.map((img, i) => (
            <div
              key={img.id}
              className={`${styles.stripItem} ${i === index ? styles.stripCurrent : ''}`}
              onClick={() => {
                if (i === index) return
                const next = images[i]
                const updated = images.map((im, j) => (j === index ? { ...im, annotated: boxes.length > 0 } : im))
                syncImagesAndNavigate(updated, next.id, i)
              }}
            >
              <AnnotatedThumbnail
                imageId={img.id}
                annotated={img.annotated}
                filename={img.filename}
                thumb
                className={styles.stripThumb}
                onClick={() => {}}
              >
                {img.annotated && <span className={styles.stripBadge}>✅</span>}
              </AnnotatedThumbnail>
            </div>
          ))}
        </div>
      </div>
      <div className={styles.toolbar}>
        <button type="button" onClick={goPrev} disabled={index <= 0}>← 上一张</button>
        <button type="button" onClick={goNext} disabled={index >= nTotal - 1}>下一张 →</button>
        <div className={styles.addDropdown} ref={(el) => { addDropdownRef.current = el }}>
          <button
            type="button"
            className={styles.addDropdownTrigger}
            onClick={() => setAddDropdownOpen((o) => !o)}
          >
            添加标注 ▼
          </button>
          {addDropdownOpen && (
            <div className={styles.addDropdownMenu}>
              <button type="button" onClick={() => { setDrawMode('ellipse'); setAddDropdownOpen(false) }} title="在画布上拖拽一次画出椭圆">
                椭圆（拖拽绘制）
              </button>
              <button type="button" onClick={() => { setDrawMode('rect'); setAddDropdownOpen(false) }} title="在画布上拖拽一次画出矩形">
                矩形（拖拽绘制）
              </button>
            </div>
          )}
        </div>
        <button
          type="button"
          disabled={!selectedId}
          onClick={() => { if (selectedId) { deleteBox(selectedId); setSelectedId(null) } }}
          title={selectedId ? '删除当前选中的椭圆（或按 Delete 键）' : '请先选中一个椭圆'}
        >
          删除选中
        </button>
        <button
          type="button"
          disabled={boxes.length === 0}
          onClick={async () => {
            if (!pathId || boxes.length === 0) return
            try {
              await clearWorkstationAnnotation(pathId)
              clearBoxes()
              setSelectedId(null)
              setImages((prev) => prev.map((img, i) => (i === index ? { ...img, annotated: false } : img)))
            } catch (e) {
              alert(e.message)
            }
          }}
          title="清除当前图全部标注，可重新添加椭圆"
        >
          清除全部
        </button>
        <button type="button" className={styles.btnSave} onClick={handleSave} disabled={saving}>
          {saving ? '保存中…' : '确认 · 保存'}
        </button>
      </div>
      <main className={styles.canvasArea}>
        <div className={styles.panel}>
          <p className={styles.panelTitle}>
            📐 操作区
            {drawMode ? ` · 请在画布上拖拽一次绘制${drawMode === 'ellipse' ? '椭圆' : '矩形'}` : '（拖拽可移动，选中后缩放，点击空白取消选中）'}
          </p>
          <Stage
            ref={stageRef}
            width={dispW}
            height={dispH}
            onMouseDown={(e) => {
              if (!drawMode) return
              const stage = e.target.getStage()
              const pos = stage.getPointerPosition()
              if (pos) {
                drawStartRef.current = { x: pos.x, y: pos.y }
                setDrawStart({ x: pos.x, y: pos.y })
                setDrawCurrent({ x: pos.x, y: pos.y })
              }
            }}
            onMouseMove={(e) => {
              if (!drawMode || !drawStartRef.current) return
              const stage = e.target.getStage()
              const pos = stage.getPointerPosition()
              if (pos) setDrawCurrent({ x: pos.x, y: pos.y })
            }}
            onMouseUp={(e) => {
              if (!drawMode || !drawStartRef.current) return
              const stage = e.target.getStage()
              const pos = stage.getPointerPosition()
              if (pos) {
                const start = drawStartRef.current
                const x1 = start.x
                const y1 = start.y
                const x2 = pos.x
                const y2 = pos.y
                const minX = Math.min(x1, x2)
                const minY = Math.min(y1, y2)
                const w = Math.max(4, Math.abs(x2 - x1))
                const h = Math.max(4, Math.abs(y2 - y1))
                const norm = toNormalized(minX, minY, w, h, dispW, dispH)
                addBox({ type: drawMode === 'ellipse' ? 'ellipse' : 'rect', ...norm, label: '' })
                drawStartRef.current = null
                setDrawMode(null)
                setDrawStart(null)
                setDrawCurrent(null)
              }
            }}
            onMouseLeave={() => {
              if (drawMode) { drawStartRef.current = null; setDrawMode(null); setDrawStart(null); setDrawCurrent(null) }
            }}
          >
            <Layer>
              <Rect
                x={0}
                y={0}
                width={dispW}
                height={dispH}
                fill="transparent"
                listening
                onClick={() => { if (!drawMode) setSelectedId(null) }}
                onTap={() => { if (!drawMode) setSelectedId(null) }}
              />
              <KonvaImageWithLoader src={imageUrl} width={dispW} height={dispH} />
              {drawStart && drawCurrent && drawMode && (
                drawMode === 'rect' ? (
                  <Rect
                    x={Math.min(drawStart.x, drawCurrent.x)}
                    y={Math.min(drawStart.y, drawCurrent.y)}
                    width={Math.max(4, Math.abs(drawCurrent.x - drawStart.x))}
                    height={Math.max(4, Math.abs(drawCurrent.y - drawStart.y))}
                    stroke="#0af"
                    strokeWidth={2}
                    dash={[4, 4]}
                    listening={false}
                  />
                ) : (
                  (() => {
                    const x1 = drawStart.x, y1 = drawStart.y, x2 = drawCurrent.x, y2 = drawCurrent.y
                    const cx = (x1 + x2) / 2
                    const cy = (y1 + y2) / 2
                    const rx = Math.max(4, Math.abs(x2 - x1) / 2)
                    const ry = Math.max(4, Math.abs(y2 - y1) / 2)
                    return <Ellipse x={cx} y={cy} radiusX={rx} radiusY={ry} stroke="#0af" strokeWidth={2} dash={[4, 4]} listening={false} />
                  })()
                )
              )}
              {boxes.map((box) => (
                <AnnotationBox
                  key={box.id}
                  box={box}
                  canvasW={dispW}
                  canvasH={dispH}
                  isSelected={box.id === selectedId}
                  onSelect={() => setSelectedId(box.id)}
                  onChange={(attrs) => updateBox(box.id, attrs)}
                  listening={!drawMode}
                />
              ))}
            </Layer>
          </Stage>
        </div>
        <div className={styles.panel}>
          <p className={styles.panelTitle}>📷 预览（实时同步）</p>
          <Stage
            width={Math.min(400, dispW)}
            height={Math.min(300, dispH)}
            scaleX={dispW > 0 ? Math.min(400, dispW) / dispW : 1}
            scaleY={dispH > 0 ? Math.min(300, dispH) / dispH : 1}
          >
            <Layer>
              <KonvaImageWithLoader src={imageUrl} width={dispW} height={dispH} />
              {boxes.map((box) => (
                <PreviewBox key={box.id} box={box} canvasW={dispW} canvasH={dispH} />
              ))}
            </Layer>
          </Stage>
        </div>
      </main>
    </div>
  )
}

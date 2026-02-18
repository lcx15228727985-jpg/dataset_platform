import { useState, useEffect, useCallback, useRef } from 'react'
import { getEpisodeImages, deleteAnnotation } from '../api/client'
import AnnotatedThumbnail from './AnnotatedThumbnail'
import styles from '../pages/Gallery.module.css'

const PAGE_SIZE = 24

export default function CursorImageGrid({
  run,
  ep,
  imageCount,
  annotatedCount,
  gridCols,
  goAnnotate,
  onDeleteAnnotation,
  onDeleted,
  collapsed = false,
  onToggle,
}) {
  const [items, setItems] = useState([])
  const [nextCursor, setNextCursor] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const containerRef = useRef(null)
  const loadingTriggerRef = useRef(null)
  const loadingRef = useRef(false)

  const handleDelete = async (imageId, e) => {
    e.stopPropagation()
    if (!window.confirm('确定移除本图标注？')) return
    try {
      await deleteAnnotation(imageId)
      setItems((prev) =>
        prev.map((i) => (i.id === imageId ? { ...i, annotated: false } : i))
      )
      onDeleted?.()
    } catch (err) {
      alert(err.message)
    }
  }

  const loadPage = useCallback(
    (cursor = null) => {
      if (loadingRef.current) return
      loadingRef.current = true
      setLoading(true)
      setError(null)
      getEpisodeImages(run, ep, cursor, PAGE_SIZE)
        .then(({ items: page, nextCursor: next }) => {
          setItems((prev) => (cursor ? [...prev, ...page] : page))
          setNextCursor(next)
        })
        .catch((e) => setError(e.message))
        .finally(() => {
          setLoading(false)
          loadingRef.current = false
        })
    },
    [run, ep]
  )

  useEffect(() => {
    if (imageCount > 0 && items.length === 0) loadPage()
  }, [imageCount, run, ep, loadPage])

  // 滚动到底部自动加载更多
  useEffect(() => {
    if (collapsed || !nextCursor || loadingRef.current) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && nextCursor && !loadingRef.current) {
          loadPage(nextCursor)
        }
      },
      { threshold: 0.1 }
    )

    if (loadingTriggerRef.current) {
      observer.observe(loadingTriggerRef.current)
    }

    return () => {
      if (loadingTriggerRef.current) {
        observer.unobserve(loadingTriggerRef.current)
      }
    }
  }, [nextCursor, collapsed, loadPage])

  if (imageCount === 0) return null

  return (
    <section className={styles.epSection}>
      <summary className={styles.epHeader} onClick={onToggle}>
        <span className={styles.epChevron}>{collapsed ? '▶' : '▼'}</span>
        <strong>{ep}</strong> · 共 {imageCount} 张 · 已标注 {annotatedCount}/{imageCount}
        {items.length < imageCount && ` · 已加载 ${items.length} 张`}
      </summary>
      {!collapsed && (
        <>
      {error && <div className={styles.error}>{error}</div>}
      <div className={styles.grid} style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)` }}>
        {items.map((img) => (
          <div key={img.id} className={styles.card}>
            <AnnotatedThumbnail
              imageId={img.id}
              annotated={img.annotated}
              filename={img.filename}
              thumb
              onClick={() => goAnnotate(img.id, items, items.findIndex((i) => i.id === img.id))}
            >
              {img.annotated && <span className={styles.badge}>✅ 已标注</span>}
            </AnnotatedThumbnail>
            <div className={styles.cardCaption}>{img.filename}</div>
            <div className={styles.cardActions}>
              <button type="button" onClick={() => goAnnotate(img.id, items, items.findIndex((i) => i.id === img.id))}>
                进入标注
              </button>
              {img.annotated && (
                <button type="button" className={styles.btnCancel} onClick={(e) => handleDelete(img.id, e)}>
                  取消标注
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
      {loading && <div className={styles.loading}>加载中…</div>}
      {/* 滚动触发器：当这个元素进入视口时自动加载 */}
      {nextCursor && !loading && (
        <div ref={loadingTriggerRef} style={{ height: '20px', marginTop: '10px' }} />
      )}
      {/* 保留手动加载按钮作为备选 */}
      {nextCursor && !loading && (
        <button type="button" onClick={() => loadPage(nextCursor)} className={styles.loadMore}>
          手动加载更多
        </button>
      )}
        </>
      )}
    </section>
  )
}

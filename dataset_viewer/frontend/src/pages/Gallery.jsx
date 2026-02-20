import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getRuns, getEpisodes, getImageUrl, deleteAnnotation, downloadExportAnnotatedImagesZip, getDeployStatus } from '../api/client'
import CursorImageGrid from '../components/CursorImageGrid'
import AnnotatedThumbnail from '../components/AnnotatedThumbnail'
import LoginButton from '../components/LoginButton'
import { sortEpisodes } from '../components/EpPieChart'
import styles from './Gallery.module.css'

export default function Gallery() {
  const [runs, setRuns] = useState([])
  const [run, setRun] = useState('')
  const [episodes, setEpisodes] = useState([])
  const [cursorAvailable, setCursorAvailable] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [deployInfo, setDeployInfo] = useState(null)
  const [gridCols, setGridCols] = useState(4)
  const [maxShow, setMaxShow] = useState(48) // 与后端 MAX_PAGE_SIZE 一致，避免单次加载过多导致 CPU 飙升
  const [collapsedEps, setCollapsedEps] = useState(() => new Set())
  const [lastAnnotated, setLastAnnotated] = useState(null)
  const navigate = useNavigate()
  
  // 读取上次标注位置
  useEffect(() => {
    try {
      const saved = localStorage.getItem('lastAnnotated')
      if (saved) {
        const parsed = JSON.parse(saved)
        setLastAnnotated(parsed)
      }
    } catch (e) {
      console.warn('Failed to load last annotated position:', e)
    }
  }, [])

  const toggleEp = (name) => {
    setCollapsedEps((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }
  const expandAll = () => setCollapsedEps(new Set())
  const collapseAll = () => setCollapsedEps(new Set(episodes.map((e) => e.name)))

  useEffect(() => {
    Promise.all([getRuns(), getDeployStatus().catch(() => null)])
      .then(([runsRes, deployRes]) => {
        setRuns(runsRes.runs || [])
        if (runsRes.runs?.length) setRun(runsRes.runs[0])
        setDeployInfo(deployRes)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!run) return
    setLoading(true)
    getEpisodes(run)
      .then((d) => {
        const eps = d.episodes || []
        setEpisodes(sortEpisodes(eps))
        setCursorAvailable(!!d.cursorAvailable)
        // 游标模式下默认折叠所有 ep，避免初次加载大量图片导致卡顿
        if (d.cursorAvailable) {
          setCollapsedEps(new Set(eps.map((e) => e.name)))
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [run])

  const goAnnotate = (imageId, images, index) => {
    const search = `?path=${encodeURIComponent(imageId)}&run=${encodeURIComponent(run)}&index=${index}`
    navigate(`/annotate${search}`, { state: { imageId, images, index, run } })
  }

  // 定位到上次标注的位置
  const goToLastAnnotated = () => {
    if (!lastAnnotated) {
      alert('暂无上次标注记录')
      return
    }
    
    const { run: lastRun, pathId, index: lastIndex } = lastAnnotated
    
    // 如果当前 run 不是上次的 run，先切换 run（episodes 会自动加载）
    if (lastRun && lastRun !== run && runs.includes(lastRun)) {
      setRun(lastRun)
      // 等待一个 tick 让 episodes 开始加载，然后直接跳转（Workbench 会处理加载）
      setTimeout(() => {
        const search = `?path=${encodeURIComponent(pathId)}&run=${encodeURIComponent(lastRun)}&index=${lastIndex || 0}`
        navigate(`/annotate${search}`, { state: { imageId: pathId, images: [], index: lastIndex || 0, run: lastRun } })
      }, 100)
    } else {
      // 直接跳转到标注页面
      const search = `?path=${encodeURIComponent(pathId)}&run=${encodeURIComponent(lastRun || run)}&index=${lastIndex || 0}`
      navigate(`/annotate${search}`, { state: { imageId: pathId, images: [], index: lastIndex || 0, run: lastRun || run } })
    }
  }

  const handleDeleteAnnotation = async (imageId, e) => {
    e.stopPropagation()
    if (!window.confirm('确定移除本图标注？')) return
    try {
      await deleteAnnotation(imageId)
      setEpisodes((prev) =>
        prev.map((ep) => {
          const hasThis = ep.images?.some((i) => i.id === imageId)
          if (!hasThis) return ep
          return {
            ...ep,
            images: ep.images.map((img) =>
              img.id === imageId ? { ...img, annotated: false } : img
            ),
            annotatedCount: Math.max(0, (ep.annotatedCount || 0) - 1),
          }
        })
      )
    } catch (err) {
      alert(err.message)
    }
  }

  if (error) return <div className={styles.error}>错误: {error}</div>
  if (loading && !runs.length) return <div className={styles.loading}>加载中…</div>

  // 数据选择为空：未建库或库中无 run，提示先初始化数据库
  if (!loading && runs.length === 0) {
    const dataRoot = deployInfo?.data_root ?? '(未获取)'
    const dataRootExists = deployInfo?.data_root_exists
    return (
      <div className={styles.page}>
        <header className={styles.header}>
          <h1>数据集标注平台</h1>
        </header>
        <main className={styles.main} style={{ padding: '2rem', textAlign: 'center' }}>
          <p className={styles.caption}>当前暂无 Run 数据。</p>
          <p className={styles.caption}>
            请先在服务器执行：<code>python -m backend.scan_dataset</code>（本地推荐 SQLite），<br />
            并确保 .env 中 <strong>DATA_ROOT</strong> 指向包含 run/episode_0/images 或 images_png 的数据目录。
          </p>
          <p className={styles.caption} style={{ fontSize: '0.9em', opacity: 0.9 }}>
            后端当前 DATA_ROOT：<code>{dataRoot}</code>，目录存在：{dataRootExists === true ? '是' : dataRootExists === false ? '否' : '未知'}
          </p>
          <p className={styles.caption}>若使用 Docker 部署，请挂载数据集目录并运行 init 容器或见部署说明。</p>
        </main>
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>数据集标注平台</h1>
          <p className={styles.caption}>数据目录由后端配置 · 点击「进入标注」进入工作台 · 登录后保存才写入数据库</p>
        </div>
        <LoginButton />
      </header>

      <aside className={styles.sidebar}>
        <h2>数据选择</h2>
        <select value={run} onChange={(e) => setRun(e.target.value)}>
          {runs.map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>

        <h2 className={styles.pieSectionTitle}>标注看板</h2>
        <button type="button" className={styles.dashboardBtn} onClick={() => navigate('/dashboard')}>
          查看标注统计 →
        </button>
        
        {lastAnnotated && (
          <>
            <h2>快速定位</h2>
            <button 
              type="button" 
              className={styles.dashboardBtn} 
              onClick={goToLastAnnotated}
              title={`上次标注：${lastAnnotated.run || '未知'} · ${new Date(lastAnnotated.timestamp).toLocaleString()}`}
            >
              📍 定位到上次标注
            </button>
          </>
        )}
        <h2>导出</h2>
        <button type="button" className={styles.exportBtn} onClick={() => downloadExportAnnotatedImagesZip().catch((e) => alert(e.message))} title="下载已标注图片 ZIP，结构：{run}标注版/episode_X/xxx.png">
          导出已标注图片 (ZIP)
        </button>

        <h2>显示设置</h2>
        <label>网格列数 <input type="range" min={2} max={8} value={gridCols} onChange={(e) => setGridCols(Number(e.target.value))} /> {gridCols}</label>
        <label>每 ep 最多预览 <input type="number" min={0} max={100} value={maxShow} onChange={(e) => setMaxShow(Math.min(100, Math.max(0, Number(e.target.value) || 0)))} /></label>

        <h2>Episode 折叠</h2>
        <div className={styles.foldBtns}>
          <button type="button" className={styles.btnSmall} onClick={expandAll}>全部展开</button>
          <button type="button" className={styles.btnSmall} onClick={collapseAll}>全部折叠</button>
        </div>
      </aside>

      <main className={styles.main}>
        <p className={styles.source}>当前 Run: <strong>{run}</strong></p>
        {cursorAvailable && (
          <p className={styles.caption}>已启用游标分页 + 虚拟列表（大图库加速）</p>
        )}

        {cursorAvailable
          ? episodes.map((ep) =>
              ep.imageCount > 0 ? (
                <CursorImageGrid
                  key={ep.name}
                  run={run}
                  ep={ep.name}
                  imageCount={ep.imageCount}
                  annotatedCount={ep.annotatedCount ?? 0}
                  gridCols={gridCols}
                  goAnnotate={goAnnotate}
                  onDeleteAnnotation={handleDeleteAnnotation}
                  onDeleted={() =>
                    setEpisodes((prev) =>
                      prev.map((e) =>
                        e.name === ep.name
                          ? { ...e, annotatedCount: Math.max(0, (e.annotatedCount ?? 0) - 1) }
                          : e
                      )
                    )
                  }
                  collapsed={collapsedEps.has(ep.name)}
                  onToggle={() => toggleEp(ep.name)}
                />
              ) : (
                <section key={ep.name} className={styles.epSection}>
                  <summary className={styles.epHeader} onClick={() => toggleEp(ep.name)}>
                    <span className={styles.epChevron}>{collapsedEps.has(ep.name) ? '▶' : '▼'}</span>
                    <strong>{ep.name}</strong> · 共 0 张
                  </summary>
                </section>
              )
            )
          : episodes.map((ep) => {
              const list = (ep.images || []).slice(0, maxShow || ep.images?.length)
              if (!list.length && ep.imageCount === 0) return null
              const collapsed = collapsedEps.has(ep.name)
              return (
                <section key={ep.name} className={styles.epSection}>
                  <summary className={styles.epHeader} onClick={() => toggleEp(ep.name)}>
                    <span className={styles.epChevron}>{collapsed ? '▶' : '▼'}</span>
                    <strong>{ep.name}</strong> · 共 {ep.imageCount} 张 · 已标注 {ep.annotatedCount}/{ep.imageCount}
                  </summary>
                  {!collapsed && (
                  <div className={styles.grid} style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)` }}>
                    {list.map((img) => (
                      <div key={img.id} className={styles.card}>
                        <AnnotatedThumbnail
                          imageId={img.id}
                          annotated={img.annotated}
                          filename={img.filename}
                          thumb={false}
                          onClick={() => goAnnotate(img.id, ep.images, ep.images.findIndex((i) => i.id === img.id))}
                        >
                          {img.annotated && <span className={styles.badge}>✅ 已标注</span>}
                        </AnnotatedThumbnail>
                        <div className={styles.cardCaption}>{img.filename}</div>
                        <div className={styles.cardActions}>
                          <button type="button" onClick={() => goAnnotate(img.id, ep.images, ep.images.findIndex((i) => i.id === img.id))}>
                            进入标注
                          </button>
                          {img.annotated && (
                            <button type="button" className={styles.btnCancel} onClick={(e) => handleDeleteAnnotation(img.id, e)}>
                              取消标注
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  )}
                </section>
              )
            })}

        <div className={styles.footer}>
          <p>图源优先 images_png，若无则 images。标注保存为同目录 _annot.json。</p>
        </div>
      </main>
    </div>
  )
}

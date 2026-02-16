import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getRuns, getEpisodes } from '../api/client'
import EpPieChart, { sortEpisodes } from '../components/EpPieChart'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const [runs, setRuns] = useState([])
  const [run, setRun] = useState('')
  const [episodes, setEpisodes] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getRuns()
      .then((d) => {
        const list = d.runs || []
        setRuns(list)
        if (list.length) setRun(list[0])
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!run) return
    setLoading(true)
    getEpisodes(run)
      .then((d) => setEpisodes(sortEpisodes(d.episodes || [])))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [run])

  if (error) return <div className={styles.error}>错误: {error}</div>

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link to="/" className={styles.backLink}>← 返回图库</Link>
        <h1>标注看板</h1>
      </header>

      <main className={styles.main}>
        <section className={styles.controls}>
          <label>数据选择
            <select value={run} onChange={(e) => setRun(e.target.value)}>
              {runs.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </label>
        </section>

        <section className={styles.chartSection}>
          <h2>各 ep 标注统计</h2>
          {loading ? (
            <p className={styles.loading}>加载中…</p>
          ) : (
            <div className={styles.chartWrap}>
              <EpPieChart episodes={episodes} large />
            </div>
          )}
        </section>

        <section className={styles.summary}>
          <h2>汇总</h2>
          <div className={styles.stats}>
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Run</span>
              <span className={styles.statValue}>{run || '-'}</span>
            </div>
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Episodes</span>
              <span className={styles.statValue}>{episodes.length}</span>
            </div>
            <div className={styles.statItem}>
              <span className={styles.statLabel}>总图片数</span>
              <span className={styles.statValue}>{episodes.reduce((s, e) => s + (e.imageCount || 0), 0).toLocaleString()}</span>
            </div>
            <div className={styles.statItem}>
              <span className={styles.statLabel}>已标注</span>
              <span className={styles.statValue}>{episodes.reduce((s, e) => s + (e.annotatedCount || 0), 0).toLocaleString()}</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

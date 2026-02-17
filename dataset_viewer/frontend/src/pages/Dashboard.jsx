import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getRuns, getEpisodes, getDashboardUserStats } from '../api/client'
import { useAuthStore } from '../store/auth'
import EpPieChart, { sortEpisodes } from '../components/EpPieChart'
import LoginButton from '../components/LoginButton'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const [runs, setRuns] = useState([])
  const [run, setRun] = useState('')
  const [episodes, setEpisodes] = useState([])
  const [userStats, setUserStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const userId = useAuthStore((s) => s.userId)

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
    Promise.all([
      getEpisodes(run),
      getDashboardUserStats(run).catch(() => null),
    ])
      .then(([epsRes, statsRes]) => {
        setEpisodes(sortEpisodes(epsRes.episodes || []))
        setUserStats(statsRes)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [run])

  if (error) return <div className={styles.error}>错误: {error}</div>

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link to="/" className={styles.backLink}>← 返回图库</Link>
        <h1>标注看板</h1>
        <LoginButton />
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
          {userStats && (
            <div className={styles.userStats}>
              <h3>按用户统计</h3>
              <div className={styles.stats}>
                <div className={styles.statItem}>
                  <span className={styles.statLabel}>当前用户 ({userId || '未登录'})</span>
                  <span className={styles.statValue}>{userStats.current_user_count?.toLocaleString() ?? 0}</span>
                </div>
                {userStats.others?.map((o) => (
                  <div key={o.user_id} className={styles.statItem}>
                    <span className={styles.statLabel}>用户 {o.user_id}</span>
                    <span className={styles.statValue}>{o.count?.toLocaleString() ?? 0}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

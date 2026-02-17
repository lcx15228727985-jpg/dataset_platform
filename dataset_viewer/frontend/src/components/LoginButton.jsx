/**
 * 简易登录：10 用户 000-009，默认密码 123456。
 * 放在主页右上角，登录后显示当前用户。
 */
import { useState } from 'react'
import { login as apiLogin } from '../api/client'
import { useAuthStore } from '../store/auth'
import styles from './LoginButton.module.css'

export default function LoginButton() {
  const { token, userId, login, logout } = useAuthStore()
  const [open, setOpen] = useState(false)
  const [uid, setUid] = useState('000')
  const [pwd, setPwd] = useState('123456')
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setErr('')
    setLoading(true)
    try {
      const { token: t, user_id } = await apiLogin(uid.trim(), pwd)
      login(t, user_id)
      setOpen(false)
    } catch (e) {
      setErr(e.message || '登录失败')
    } finally {
      setLoading(false)
    }
  }

  if (token && userId) {
    return (
      <div className={styles.wrap}>
        <span className={styles.user}>用户 {userId}</span>
        <button type="button" className={styles.logoutBtn} onClick={() => logout()}>
          退出
        </button>
      </div>
    )
  }

  return (
    <div className={styles.wrap}>
      <button type="button" className={styles.loginBtn} onClick={() => setOpen(!open)}>
        登录
      </button>
      {open && (
        <div className={styles.dropdown}>
          <form onSubmit={handleSubmit}>
            <p className={styles.hint}>账号 000-009，密码 123456</p>
            <input
              type="text"
              placeholder="账号"
              value={uid}
              onChange={(e) => setUid(e.target.value)}
              maxLength={3}
              autoFocus
            />
            <input
              type="password"
              placeholder="密码"
              value={pwd}
              onChange={(e) => setPwd(e.target.value)}
            />
            {err && <p className={styles.err}>{err}</p>}
            <button type="submit" disabled={loading}>
              {loading ? '登录中…' : '确定'}
            </button>
          </form>
        </div>
      )}
    </div>
  )
}

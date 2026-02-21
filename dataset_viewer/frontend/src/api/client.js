import { useAuthStore } from '../store/auth'

const API_BASE = import.meta.env.VITE_API_URL || '';

function url(path) {
  return `${API_BASE}${path}`;
}

function authHeaders() {
  const token = useAuthStore.getState().token
  if (!token) return {}
  return { Authorization: `Bearer ${token}` }
}

/** 看板按用户统计：当前用户及其他用户标注数量 */
export async function getDashboardUserStats(run) {
  const r = await fetch(url(`/api/dashboard/user-stats?run=${encodeURIComponent(run)}`), {
    headers: authHeaders(),
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

/** 部署状态：data_root、database 等，用于排查「暂无 Run 数据」 */
export async function getDeployStatus() {
  const r = await fetch(url('/api/deploy/status'));
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getRuns() {
  const r = await fetch(url('/api/runs'));
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getEpisodes(run) {
  const r = await fetch(url(`/api/runs/${encodeURIComponent(run)}/episodes`));
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function getImageUrl(imageId, thumb = false) {
  const thumbParam = thumb ? '&thumb=1' : ''
  return url(`/api/image?path=${encodeURIComponent(imageId)}${thumbParam}`)
}

/** 游标分页：拉取某 episode 下的一页图片（需后端已建 catalog DB） */
export async function getEpisodeImages(run, ep, cursor = null, limit = 50) {
  const params = new URLSearchParams()
  if (cursor) params.set('cursor', cursor)
  params.set('limit', String(limit))
  const r = await fetch(url(`/api/runs/${encodeURIComponent(run)}/episodes/${encodeURIComponent(ep)}/images?${params}`))
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getAnnotation(imageId) {
  const r = await fetch(url(`/api/annotation/${encodeURIComponent(imageId)}`));
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function saveAnnotation(imageId, data) {
  const r = await fetch(url(`/api/annotation/${encodeURIComponent(imageId)}`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/** 工作台：获取单张图的数据（path、尺寸、boxes） */
export async function getWorkstationData(pathId) {
  const r = await fetch(url(`/api/workstation/data?path=${encodeURIComponent(pathId)}`))
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

/** 登录：user_id 000-009，密码 123456 */
export async function login(userId, password) {
  const r = await fetch(url('/api/auth/login'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, password }),
  })
  if (!r.ok) {
    const text = await r.text()
    let msg = text
    try {
      const j = JSON.parse(text)
      if (j.detail) msg = j.detail
    } catch (_) {}
    throw new Error(msg)
  }
  return r.json()
}

/** 验证当前 token */
export async function authMe() {
  const r = await fetch(url('/api/auth/me'), {
    headers: authHeaders(),
  })
  if (!r.ok) throw new Error('未登录或 token 已过期')
  return r.json()
}

/** 工作台：清除当前图全部标注（与主页取消标注同步）。需登录。 */
export async function clearWorkstationAnnotation(pathId) {
  const r = await fetch(url('/api/workstation/save'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ path_id: pathId, boxes: [] }),
  })
  if (!r.ok) {
    const text = await r.text()
    try {
      const j = JSON.parse(text)
      throw new Error(j.detail || text)
    } catch (e) {
      if (e instanceof SyntaxError) throw new Error(text)
      throw e
    }
  }
  return r.json()
}

/** 工作台：保存 boxes。需登录，否则 401。 */
export async function saveWorkstationBoxes(pathId, boxes) {
  const r = await fetch(url('/api/workstation/save'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ path_id: pathId, boxes }),
  })
  if (!r.ok) {
    const text = await r.text()
    try {
      const j = JSON.parse(text)
      throw new Error(j.detail || text)
    } catch (e) {
      if (e instanceof SyntaxError) throw new Error(text)
      throw e
    }
  }
  return r.json()
}

/** 导出已标注数据：run、ep、path_id、filename、boxes。format=json|csv，save=1 时同时保存到服务器 */
export async function downloadExportAnnotations(format = 'json', save = false) {
  const params = new URLSearchParams({ format })
  if (save) params.set('save', '1')
  const r = await fetch(url(`/api/export/annotations?${params}`))
  if (!r.ok) throw new Error(await r.text())
  const blob = await r.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = format === 'csv' ? 'annotations.csv' : 'annotations.json'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(a.href)
}

/** 导出已标注图片为 ZIP，目录结构：{run}标注版/episode_X/xxx.png */
export async function downloadExportAnnotatedImagesZip() {
  const r = await fetch(url('/api/export/annotated-images'))
  if (!r.ok) {
    const text = await r.text()
    let msg = text
    try {
      const j = JSON.parse(text)
      if (j.detail) msg = j.detail
    } catch (_) {}
    throw new Error(msg)
  }
  const blob = await r.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = 'annotated_images.zip'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(a.href)
}

export async function deleteAnnotation(imageId) {
  const r = await fetch(url(`/api/annotation/${encodeURIComponent(imageId)}`), { method: 'DELETE' });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

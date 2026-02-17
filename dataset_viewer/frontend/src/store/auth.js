/**
 * 认证状态：10 用户 000-009，默认密码 123456。
 * 登录后 token 存入 localStorage，供 API 请求携带。
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const STORAGE_KEY = 'dataset-annotation-auth'

export const useAuthStore = create(
  persist(
    (set) => ({
      token: null,
      userId: null,
      login: (token, userId) => set({ token, userId }),
      logout: () => set({ token: null, userId: null }),
      isLoggedIn: () => {
        const state = useAuthStore.getState()
        return !!(state.token && state.userId)
      },
    }),
    { name: STORAGE_KEY }
  )
)

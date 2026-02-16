import { create } from 'zustand'

export const useWorkbenchStore = create((set, get) => ({
  pathId: null,
  imageUrl: null,
  meta: { width: 0, height: 0 },
  boxes: [],

  loadImage: (data) => set({
    pathId: data.path_id,
    imageUrl: data.imageUrl ?? null,
    meta: data.meta ?? { width: 0, height: 0 },
    boxes: Array.isArray(data.boxes) ? data.boxes : [],
  }),

  addBox: (newBox) => set((state) => ({
    boxes: [...state.boxes, { ...newBox, id: newBox.id || `box_${Date.now()}_${Math.random().toString(36).slice(2)}` }],
  })),

  updateBox: (id, attrs) => set((state) => ({
    boxes: state.boxes.map((b) => (b.id === id ? { ...b, ...attrs } : b)),
  })),

  deleteBox: (id) => set((state) => ({
    boxes: state.boxes.filter((b) => b.id !== id),
  })),

  clearBoxes: () => set({ boxes: [] }),

  clear: () => set({
    pathId: null,
    imageUrl: null,
    meta: { width: 0, height: 0 },
    boxes: [],
  }),
}))

# 图标库 Icons

本目录用于存放 UI 图标的 SVG 文件，方便项目内统一引用。

## 使用方式

```jsx
import Icon from '../components/Icon'

// 按名称引用
<Icon name="arrow-left" size={20} />
<Icon name="save" className={styles.myIcon} />
```

## 添加新图标

1. 将 `.svg` 或 `.png` 文件放入本目录，命名为小写字母和连字符（如 `arrow-right.svg`、`arrow-right.png`）
2. 在组件中通过 `name="arrow-right"` 引用（不需写扩展名）
3. 同名的 `.svg` 优先于 `.png` 加载

## 图标命名建议

- 左箭头：`arrow-left`
- 右箭头：`arrow-right`
- 保存：`save`
- 删除：`trash` / `delete`
- 清除：`clear` / `eraser`
- 椭圆：`ellipse` / `circle`
- 矩形：`rect` / `square`
- 下拉箭头：`chevron-down`

## 格式建议

- 使用 24×24 或 16×16 viewBox，便于统一缩放
- 保持 SVG 简洁，可被 `currentColor`  inherit 的用 `fill="currentColor"`
- 线宽建议 1.5–2，保证小尺寸可读

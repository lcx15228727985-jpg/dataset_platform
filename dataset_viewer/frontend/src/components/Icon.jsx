/**
 * 图标组件 - 从 assets/icons 按名称加载 SVG
 * 用法：<Icon name="arrow-left" size={20} />
 */
import React from 'react'

const svgModules = import.meta.glob('../assets/icons/*.svg', {
  query: '?url',
  import: 'default',
  eager: true,
})
const pngModules = import.meta.glob('../assets/icons/*.png', {
  query: '?url',
  import: 'default',
  eager: true,
})

function getIconUrl(name) {
  const svgKey = Object.keys(svgModules).find((k) => k.endsWith(`/${name}.svg`))
  if (svgKey) return svgModules[svgKey]
  const pngKey = Object.keys(pngModules).find((k) => k.endsWith(`/${name}.png`))
  if (pngKey) return pngModules[pngKey]
  return null
}

export default function Icon({ name, size = 24, width, height, className, alt = '', ...rest }) {
  const src = getIconUrl(name)
  if (!src) return null
  const w = width ?? size
  const h = height ?? size
  return (
    <img
      src={src}
      alt={alt}
      width={w}
      height={h}
      className={className}
      aria-hidden={!alt}
      {...rest}
    />
  )
}

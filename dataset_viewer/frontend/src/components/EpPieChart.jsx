import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

const COLORS = ['#0a5', '#08c', '#c80', '#a50', '#70a', '#a07', '#07a', '#7a0', '#07c', '#c07', '#a70', '#0a7', '#7c0']

/** ep 名称转数字用于排序 */
export function epSortKey(name) {
  const m = /^ep(\d+)$/i.exec(name)
  return m ? parseInt(m[1], 10) : 999
}

/** 将 episodes 按 ep0, ep1, ..., ep12 数字顺序排序 */
export function sortEpisodes(episodes) {
  return [...(episodes || [])].sort((a, b) => epSortKey(a.name) - epSortKey(b.name))
}

/** 饼图：各 ep 标注情况（slice 大小为 imageCount，显示已标注/总数）。large 时放大用于看板页 */
export default function EpPieChart({ episodes, large = false }) {
  const sorted = sortEpisodes(episodes)
  const data = sorted.map((ep, i) => ({
    name: ep.name,
    value: ep.imageCount || 0,
    annotated: ep.annotatedCount ?? 0,
    total: ep.imageCount || 0,
    fill: COLORS[i % COLORS.length],
  })).filter(d => d.value > 0)

  if (data.length === 0) return <p className="chartEmpty">暂无数据</p>

  const chartH = large ? 520 : 220
  const innerR = large ? 60 : 36
  const outerR = large ? 100 : 58

  return (
    <div className="epPieChartWrap">
      <ResponsiveContainer width="100%" height={chartH}>
        <PieChart margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="38%"
            innerRadius={innerR}
            outerRadius={outerR}
            paddingAngle={2}
            label={({ name }) => name}
          >
            {data.map((entry, i) => (
              <Cell key={entry.name} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value, name, entry) => [`已标注 ${entry?.payload?.annotated ?? 0}/${entry?.payload?.total ?? 0}`, name]}
          />
          <Legend
            layout="vertical"
            verticalAlign="bottom"
            align="center"
            wrapperStyle={{ paddingTop: 8 }}
            iconSize={6}
            iconType="circle"
            formatter={(value) => {
              const d = data.find(x => x.name === value)
              return d ? `${value}: ${d.annotated}/${d.total}` : value
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

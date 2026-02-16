import { Routes, Route } from 'react-router-dom'
import Gallery from './pages/Gallery'
import Workbench from './pages/Workbench'
import Dashboard from './pages/Dashboard'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Gallery />} />
      <Route path="/annotate" element={<Workbench />} />
      <Route path="/dashboard" element={<Dashboard />} />
    </Routes>
  )
}

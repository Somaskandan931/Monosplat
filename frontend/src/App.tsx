import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import Dashboard        from '@/pages/Dashboard'
import DatasetManager   from '@/pages/DatasetManager'
import TrainingDashboard from '@/pages/TrainingDashboard'
import Viewer from '@/pages/Viewer'


export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/"            element={<Dashboard />}         />
          <Route path="/datasets"    element={<DatasetManager />}    />
          <Route path="/training"    element={<TrainingDashboard />} />

          <Route path="/viewer"      element={<Viewer />}            />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

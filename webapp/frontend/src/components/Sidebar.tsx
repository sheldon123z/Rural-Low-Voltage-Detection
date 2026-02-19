import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Zap, History, BarChart3, BookOpen, Activity
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: '总览仪表板' },
  { path: '/detect', icon: Zap, label: '异常检测' },
  { path: '/history', icon: History, label: '检测历史' },
  { path: '/models', icon: BarChart3, label: '模型对比' },
  { path: '/about', icon: BookOpen, label: '系统原理' },
]

export function Sidebar() {
  return (
    <aside className="w-56 bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-5 border-b border-border">
        <Activity className="w-6 h-6 text-primary mr-2.5" />
        <div>
          <div className="text-sm font-semibold text-foreground leading-tight">低电压检测</div>
          <div className="text-xs text-muted-foreground">Rural Grid AI</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-1 px-3">
        {navItems.map(({ path, icon: Icon, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )
            }
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Status */}
      <div className="p-4 border-t border-border">
        <div className="text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5 mb-1">
            <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span>VoltageTimesNet 已加载</span>
          </div>
          <div className="text-muted-foreground/70">F1 = 0.8149 | Recall = 91.1%</div>
        </div>
      </div>
    </aside>
  )
}

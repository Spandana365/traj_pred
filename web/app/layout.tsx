import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { ExperimentProvider } from '@/stores/experiment-store'
import './globals.css'
export const metadata: Metadata = { title:'TrackVision · Research Workspace', description:'Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments', generator:'TrackVision' }
export const viewport: Viewport = { colorScheme:'dark', themeColor:'#070a0f' }
export default function RootLayout({children}:{children:React.ReactNode}){ return <html lang="en"><body className="antialiased"><ExperimentProvider>{children}</ExperimentProvider>{process.env.NODE_ENV==='production'&&<Analytics/>}</body></html> }

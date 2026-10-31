export const appConfig = { name:'TrackVision', eyebrow:'Research workspace', title:'Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments', subtitle:'An Integrated Research Platform for Pipeline Error Propagation Analysis', apiBaseUrl: process.env.NEXT_PUBLIC_TRACKVISION_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000', websocketUrl: process.env.NEXT_PUBLIC_WS_URL ?? '' }

export const navItems = [
  { label:'Dashboard', href:'/overview' },
  { label:'Experiments', href:'/experiments' },
  { label:'Workflow', href:'/overview' },
  { label:'Results', href:'/analysis' },
  { label:'Settings', href:'/settings' },
]

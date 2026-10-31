export type SourceType = 'dataset' | 'uploaded-video'
export type GroundTruthStatus = 'available' | 'unavailable' | 'pending'
export type ExperimentStatus = 'draft' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
export type Tracker = 'ByteTrack' | 'BoT-SORT'
export type PredictionModel = 'Constant Velocity' | 'Constant Acceleration' | 'LSTM' | 'GRU'
export type ErrorType = 'Localization Noise' | 'Observation Dropout' | 'Track Fragmentation' | 'Identity Switching' | 'Ego-Motion Error'
export type WorkflowStage = 'overview' | 'data' | 'tracking' | 'error-injection' | 'error-visualization' | 'prediction' | 'analysis' | 'application-impact'
export type ArtifactType = 'video' | 'tracking_result' | 'trajectory' | 'prediction' | 'error_visualization' | 'chart' | 'report' | 'model'

export interface VideoMetadata { name: string; size?: string; resolution?: string; fps?: number; duration?: string; frameCount?: number }
export interface Dataset { id: string; name: string; scene: string; sequence: string; frameCount: number; fps: number; resolution: string; annotationAvailability: GroundTruthStatus; calibrationAvailability: GroundTruthStatus; regime: string }
export interface Detection { frame: number; x: number; y: number; width: number; height: number; confidence: number; className: string }
export interface Track { id: number; state: 'active' | 'lost' | 'finished'; length: number; lastFrame: number }
export interface TrajectoryPoint { x: number; y: number; timestamp?: number; frame: number; trackId: number; trajectoryType: 'ground-truth' | 'observation' | 'corrupted' | 'prediction' }
export interface Trajectory { trackId: number; points: TrajectoryPoint[]; groundTruthAvailable: boolean }
export interface TrackingMetrics { mota: number | null; idf1: number | null; hota: number | null; idSwitches: number | null; fragmentations: number | null }
export interface TrackingResult { status: ExperimentStatus; activeTracks: number | null; totalDetections: number | null; averageTrackLength: number | null; fps: number | null; latency: number | null; metrics: TrackingMetrics | null; tracks: Track[] }
export interface ErrorConfiguration { type: ErrorType; enabled: boolean; severity: number; parameters: Record<string, string | number | boolean>; mode: 'single' | 'combined' }
export interface ErrorResult { status: ExperimentStatus; requestedSeverity: number | null; achievedSeverity: number | null; affectedTracks: number | null; affectedFrames: number | null; events: Array<{ frame: number; label: string }> }
export interface PredictionConfiguration { models: PredictionModel[]; historyLength: number; horizonLength: number; selectedTrack: number | null }
export interface PredictionMetrics { ade: number | null; fde: number | null; rmse: number | null; perStepError: number[]; coverage: number | null }
export interface PredictionResult { status: ExperimentStatus; modelVersion: string | null; metrics: PredictionMetrics | null }
export interface Artifact { id: string; type: ArtifactType; name: string; path?: string; createdAt: string; experimentId: string; status: ExperimentStatus }
export interface ExperimentFilter { dataset?: string; scene?: string; regime?: string; tracker?: Tracker; predictor?: PredictionModel; errorType?: ErrorType; severity?: string; historyLength?: string; horizonLength?: string; experiment?: string }
export interface AnalysisResult { hasData: boolean; isMock: boolean; series: Array<Record<string, string | number>> }
export interface ApplicationImpactResult { hasData: boolean; isMock: boolean; clean: Record<string, number | null>; errorControlled: Record<string, number | null> }
export interface Experiment { experimentId: string; timestamp: string; sourceType: SourceType; sourceName: string; sourceHash: string | null; dataset: string | null; scene: string | null; regime: string | null; groundTruthAvailable: boolean; detector: string; detectorVersion: string | null; detectorParameters: Record<string, unknown>; tracker: Tracker | null; trackerVersion: string | null; trackerParameters: Record<string, unknown>; stabilizer: string | null; stabilizerVersion: string | null; predictor: PredictionModel | null; predictorVersion: string | null; predictorParameters: Record<string, unknown>; historyLength: number | null; horizonLength: number | null; errorType: ErrorType | null; errorParameters: Record<string, unknown>; requestedSeverity: number | null; achievedSeverity: number | null; mitigation: string | null; seed: number | null; status: ExperimentStatus; artifacts: Artifact[]; results: { tracking?: TrackingResult; error?: ErrorResult; prediction?: PredictionResult; analysis?: AnalysisResult; impact?: ApplicationImpactResult } }
export interface ExperimentState { experiment: Experiment; activeStage: WorkflowStage; video: VideoMetadata | null; sourceType: SourceType; groundTruthStatus: GroundTruthStatus; tracking: TrackingResult | null; errorConfiguration: ErrorConfiguration | null; errorResult: ErrorResult | null; predictionConfiguration: PredictionConfiguration; prediction: PredictionResult | null; filters: ExperimentFilter }
export interface ApiResponse<T> { data: T | null; status: 'success' | 'empty' | 'error'; message?: string }
export interface JobProgressState { status: ExperimentStatus; progress: number | null; currentStage: string | null; elapsedSeconds: number | null }
export interface Stage { id: WorkflowStage; number: string; label: string; shortLabel: string; href: string }
export const workflowStages: Stage[] = [
  { id:'overview', number:'01', label:'Overview', shortLabel:'Overview', href:'/overview' },
  { id:'data', number:'02', label:'Data & Ground Truth', shortLabel:'Data', href:'/data' },
  { id:'tracking', number:'03', label:'Tracking', shortLabel:'Tracking', href:'/tracking' },
  { id:'error-injection', number:'04', label:'Error Injection', shortLabel:'Error', href:'/error-injection' },
  { id:'error-visualization', number:'05', label:'Error Visualization', shortLabel:'Visualize', href:'/error-visualization' },
  { id:'prediction', number:'06', label:'Trajectory Prediction', shortLabel:'Prediction', href:'/prediction' },
  { id:'analysis', number:'07', label:'Analysis', shortLabel:'Analysis', href:'/analysis' },
  { id:'application-impact', number:'08', label:'Application Impact', shortLabel:'Impact', href:'/application-impact' },
]
export const emptyMetrics: TrackingMetrics = { mota:null, idf1:null, hota:null, idSwitches:null, fragmentations:null }
export const emptyPredictionMetrics: PredictionMetrics = { ade:null, fde:null, rmse:null, perStepError:[], coverage:null }
export const createInitialExperiment = (): Experiment => ({ experimentId:'saved-results', timestamp:'', sourceType:'dataset', sourceName:'Saved research results', sourceHash:null, dataset:null, scene:null, regime:null, groundTruthAvailable:false, detector:'', detectorVersion:null, detectorParameters:{}, tracker:null, trackerVersion:null, trackerParameters:{}, stabilizer:null, stabilizerVersion:null, predictor:null, predictorVersion:null, predictorParameters:{}, historyLength:null, horizonLength:null, errorType:null, errorParameters:{}, requestedSeverity:null, achievedSeverity:null, mitigation:null, seed:null, status:'draft', artifacts:[], results:{} })
export const createInitialState = (): ExperimentState => ({ experiment:createInitialExperiment(), activeStage:'overview', video:null, sourceType:'dataset', groundTruthStatus:'available', tracking:null, errorConfiguration:null, errorResult:null, predictionConfiguration:{ models:['Constant Velocity'], historyLength:12, horizonLength:12, selectedTrack:null }, prediction:null, filters:{} })
export const isQuantitativeEvaluationAvailable = (state: ExperimentState) => state.groundTruthStatus === 'available'
export const isPredictionReady = (state: ExperimentState) => Boolean(state.tracking)
export const isImpactReady = (state: ExperimentState) => Boolean(state.prediction?.metrics)
export const formatMetric = (value: number | null) => value === null ? '—' : value.toFixed(3)

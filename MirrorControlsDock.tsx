import { useRef, useState } from 'react';
import {
  Camera,
  ChevronDown,
  CloudFog,
  Eraser,
  Film,
  ImagePlus,
  ListMusic,
  Loader2,
  Mic,
  MicOff,
  Music,
  Pause,
  PenTool,
  Play,
  RefreshCw,
  Repeat,
  RotateCcw,
  RotateCw,
  Sparkles,
  Square,
  Subtitles,
  Upload,
  Video,
} from 'lucide-react';
import type {
  AudioSourceKind,
  DetectionStage,
} from '../hooks/useMusicLyricsSync';
import type {
  FogCoveragePreset,
  FogPhase,
  FogStatus,
  LyricLine,
  MirrorSettings,
} from '../mirror/types';

type DockTab = 'music' | 'fog' | 'studio';

export type FramingGuide = 'none' | '9:16' | '4:5';

interface MusicStateProps {
  sourceKind: AudioSourceKind;
  detectionStage: DetectionStage;
  statusDetail: string;
  songTitle: string;
  bpm: number | null;
  mood: string | null;
  isAnalyzing: boolean;
  isPlaying: boolean;
  isMicListening: boolean;
  isScriptPlaying: boolean;
  hasLoadedFile: boolean;
  currentTime: number;
  duration: number;
  lyrics: LyricLine[];
  activeLineIndex: number;
  currentLyricText: string;
  hasVocals: boolean | null;
  loopTrack: boolean;
  lyricOffsetSec: number;
  errorMsg: string | null;
  onToggleLoop: () => void;
  onAdjustOffset: (delta: number) => void;
  onUploadFile: (file: File) => void;
  onRetryAnalysis: () => void;
  onAlignPlainLyrics: (plainText: string) => void;
  onImportLyricsFile: (file: File) => void;
  onTogglePlayPause: () => void;
  onSeek: (sec: number) => void;
  onJumpToLyric: (idx: number) => void;
  onToggleMic: () => void;
  onWriteManualLine: (text: string) => void;
  onStartTextQueue: (rawBlock: string) => void;
  onStopTextQueue: () => void;
}

interface RecordingStateProps {
  isRecording: boolean;
  recordingSec: number;
  framingGuide: FramingGuide;
  hasCustomMedia: boolean;
  customMediaName: string | null;
  onToggleRecording: () => void;
  onChangeFramingGuide: (guide: FramingGuide) => void;
  onUploadBackdropMedia: (file: File) => void;
}

interface MirrorControlsDockProps {
  settings: MirrorSettings;
  fogStatus: FogStatus;
  cameraError: string | null;
  collapsed: boolean;
  music: MusicStateProps;
  recording: RecordingStateProps;
  onToggleCollapse: () => void;
  onChangeSettings: (patch: Partial<MirrorSettings>) => void;
  onSteamBurst: () => void;
  onClearFog: () => void;
}

const PHASE_LABELS: Record<FogPhase, string> = {
  fogging: '持續凝霧中',
  saturated: '霧氣飽和',
  clearing: '輕微換氣',
  clear: '鏡面清澈',
};

const COVERAGE_PRESETS: {
  id: FogCoveragePreset;
  label: string;
  value: number;
}[] = [
  { id: 'edge', label: '僅邊緣', value: 0.28 },
  { id: 'halo', label: '環狀', value: 0.58 },
  { id: 'full', label: '全鏡面', value: 1.0 },
];

function formatTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, '0')}`;
}

export function MirrorControlsDock({
  settings,
  fogStatus,
  cameraError,
  collapsed,
  music,
  recording,
  onToggleCollapse,
  onChangeSettings,
  onSteamBurst,
  onClearFog,
}: MirrorControlsDockProps) {
  const [activeTab, setActiveTab] = useState<DockTab>('music');
  const [manualText, setManualText] = useState('');
  const [showScriptDrawer, setShowScriptDrawer] = useState(false);
  const [scriptText, setScriptText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const subtitleInputRef = useRef<HTMLInputElement>(null);
  const mediaInputRef = useRef<HTMLInputElement>(null);

  const fogPercent = Math.round(fogStatus.fogLevel * 100);
  const densityPercent = Math.round(settings.maxDensity * 100);
  const coveragePercent = Math.round(settings.fogCoverage * 100);
  const deltaTempC = Math.round(4 + settings.tempDiff * 16);

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!manualText.trim()) return;
    music.onWriteManualLine(manualText);
    setManualText('');
  };

  const isCameraActive = settings.mirrorSource === 'camera';
  const isMediaActive = settings.mirrorSource === 'media';

  return (
    <div
      className={`fixed bottom-2.5 sm:bottom-4 left-1/2 -translate-x-1/2 z-20 w-[min(960px,95vw)] transition-all duration-300 ease-out pointer-events-none ${
        collapsed
          ? 'translate-y-[calc(100%+28px)] opacity-0'
          : 'translate-y-0 opacity-100'
      }`}
    >
      {/* Single Clean Self-Contained Control Dock (Zero Protruding Top Box) */}
      <div className="pointer-events-auto bg-[#11141a]/95 border border-white/14 rounded-xl px-3 py-2.5 sm:px-5 sm:py-3.5 shadow-[0_22px_58px_rgba(0,0,0,0.78)] space-y-2.5 max-h-[68dvh] overflow-y-auto">
        {/* Dock Top Row: Clean 3-column non-overflowing tabs + inline Hide button */}
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-white/8">
          <div className="flex items-center gap-1.5 w-full sm:w-auto">
            <div className="grid grid-cols-3 flex-1 sm:flex-initial gap-1 p-0.5 bg-[#0a0c11] border border-white/10 rounded-lg">
              <button
                type="button"
                onClick={() => setActiveTab('music')}
                className={`inline-flex items-center justify-center gap-1 px-2 sm:px-3 py-1.5 text-[11px] sm:text-xs font-medium rounded-md transition-colors cursor-pointer whitespace-nowrap ${
                  activeTab === 'music'
                    ? 'bg-[#e6eef7] text-[#0d1016]'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                <Music className="w-3.5 h-3.5 shrink-0" />
                <span>歌詞手寫</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('fog')}
                className={`inline-flex items-center justify-center gap-1 px-2 sm:px-3 py-1.5 text-[11px] sm:text-xs font-medium rounded-md transition-colors cursor-pointer whitespace-nowrap ${
                  activeTab === 'fog'
                    ? 'bg-[#e6eef7] text-[#0d1016]'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                <CloudFog className="w-3.5 h-3.5 shrink-0" />
                <span>霧氣與筆畫</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('studio')}
                className={`inline-flex items-center justify-center gap-1 px-2 sm:px-3 py-1.5 text-[11px] sm:text-xs font-medium rounded-md transition-colors cursor-pointer whitespace-nowrap ${
                  activeTab === 'studio'
                    ? 'bg-[#e6eef7] text-[#0d1016]'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                <Film className="w-3.5 h-3.5 shrink-0" />
                <span>底景與錄影</span>
              </button>
            </div>

            <button
              type="button"
              onClick={onToggleCollapse}
              title="隱藏面板（雙擊鏡面可恢復）"
              aria-label="收合控制面板進入純淨錄影模式"
              className="inline-flex sm:hidden items-center justify-center w-8 h-8 rounded-lg bg-[#1c2029] hover:bg-[#252a35] border border-white/12 text-[#c8dcf0] shrink-0 cursor-pointer"
            >
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2 w-full sm:w-auto justify-between sm:justify-end">
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={recording.onToggleRecording}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  recording.isRecording
                    ? 'bg-[#b4362c]/35 border-[#e87c72]/70 text-[#ffe1dd]'
                    : 'bg-[#1c2029] hover:bg-[#252a35] border-white/12 text-[#eef3fa]'
                }`}
              >
                {recording.isRecording ? (
                  <>
                    <Square className="w-3 h-3 fill-current" />
                    <span className="tabular-nums">
                      停止 ({formatTime(recording.recordingSec)})
                    </span>
                  </>
                ) : (
                  <>
                    <Video className="w-3.5 h-3.5 text-[#c8dcf0]" />
                    <span>錄製剪片檔</span>
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() =>
                  onChangeSettings({
                    mirrorSource: isCameraActive ? 'studio' : 'camera',
                  })
                }
                className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  isCameraActive
                    ? 'bg-[#c8dcf0]/20 border-[#c8dcf0]/50 text-[#edf5ff]'
                    : 'bg-[#1c2029] border-white/10 text-[#e8ecf2] hover:bg-[#252a35]'
                }`}
              >
                {isCameraActive ? (
                  <>
                    <Camera className="w-3.5 h-3.5 text-[#c8dcf0]" />
                    <span>實鏡中</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5 text-[#c8dcf0]" />
                    <span>前置鏡頭</span>
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={onSteamBurst}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg bg-[#1c2029] hover:bg-[#252a35] text-[#e8ecf2] border border-white/10 transition-colors cursor-pointer whitespace-nowrap shrink-0"
              >
                <CloudFog className="w-3.5 h-3.5 text-[#c8dcf0]" />
                <span>瞬時補霧</span>
              </button>

              <button
                type="button"
                onClick={onClearFog}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg bg-[#1c2029] hover:bg-[#252a35] text-[#e8ecf2] border border-white/10 transition-colors cursor-pointer whitespace-nowrap shrink-0"
              >
                <Eraser className="w-3.5 h-3.5 text-white/70" />
                <span>暫時除霧</span>
              </button>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 text-xs text-white/75 pl-1 whitespace-nowrap shrink-0">
              <span className="text-white/55">
                {PHASE_LABELS[fogStatus.phase]}
              </span>
              <span aria-hidden="true" className="text-white/30">
                ·
              </span>
              <span className="tabular-nums font-medium text-[#f3f6fa]">
                {fogPercent}%
              </span>
            </div>

            <button
              type="button"
              onClick={onToggleCollapse}
              title="隱藏面板（雙擊鏡面或按 H 鍵可恢復）"
              className="hidden sm:inline-flex items-center gap-1 px-2.5 py-1.5 text-xs text-white/75 hover:text-white bg-[#1a1e26] hover:bg-[#252a35] border border-white/10 rounded-lg cursor-pointer whitespace-nowrap shrink-0"
            >
              <ChevronDown className="w-3.5 h-3.5 text-[#c8dcf0]" />
              <span>隱藏</span>
            </button>
          </div>
        </div>

        {cameraError && (
          <p className="text-xs text-[#ffd2cc] bg-[#b4362c]/18 border border-[#e07064]/35 rounded-lg px-3 py-2 whitespace-normal break-words leading-relaxed">
            {cameraError}
          </p>
        )}

        {/* Tab 1: Music Auto-Detection & Handwritten Text Input */}
        {activeTab === 'music' && (
          <div className="space-y-2.5">
            <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_1fr] gap-2.5 items-start">
              {/* Column A: Auto-Detect Lyrics from Unlimited Audio / Live Mic */}
              <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*,video/mp4,video/quicktime,.mp3,.wav,.flac,.m4a,.aac,.ogg,.mov"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) music.onUploadFile(file);
                    e.target.value = '';
                  }}
                />
                <input
                  ref={subtitleInputRef}
                  type="file"
                  accept=".lrc,.srt,.txt"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) music.onImportLyricsFile(file);
                    e.target.value = '';
                  }}
                />
                <button
                  type="button"
                  disabled={music.isAnalyzing}
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#c8dcf0]/18 hover:bg-[#c8dcf0]/28 border border-[#c8dcf0]/45 text-[#edf5ff] transition-colors cursor-pointer disabled:opacity-55 whitespace-nowrap shrink-0"
                >
                  {music.isAnalyzing ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Upload className="w-3.5 h-3.5" />
                  )}
                  <span>
                    {music.isAnalyzing ? 'AI 聆聽對拍中...' : '上傳音樂偵測歌詞（不限大小）'}
                  </span>
                </button>

                <button
                  type="button"
                  onClick={music.onToggleMic}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                    music.isMicListening
                      ? 'bg-[#c94a3c]/25 border-[#e07064]/60 text-[#ffd3cd]'
                      : 'bg-[#1c2029] hover:bg-[#252a35] border-white/10 text-[#e8ecf2]'
                  }`}
                >
                  {music.isMicListening ? (
                    <>
                      <MicOff className="w-3.5 h-3.5" />
                      <span>停止聽歌</span>
                    </>
                  ) : (
                    <>
                      <Mic className="w-3.5 h-3.5 text-[#c8dcf0]" />
                      <span>即時聽歌</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => subtitleInputRef.current?.click()}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg bg-[#1c2029] hover:bg-[#252a35] border border-white/10 text-[#e8ecf2] transition-colors cursor-pointer whitespace-nowrap shrink-0"
                >
                  <Subtitles className="w-3.5 h-3.5 text-[#c8dcf0]" />
                  <span>LRC/SRT</span>
                </button>

                <button
                  type="button"
                  onClick={() => setShowScriptDrawer((v) => !v)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                    showScriptDrawer || music.isScriptPlaying
                      ? 'bg-[#c8dcf0]/18 border-[#c8dcf0]/45 text-[#edf5ff]'
                      : 'bg-[#1c2029] hover:bg-[#252a35] border-white/10 text-white/75 hover:text-white'
                  }`}
                >
                  <ListMusic className="w-3.5 h-3.5 text-[#c8dcf0]" />
                  <span>多句輪寫 / 對齊歌詞</span>
                </button>
              </div>

              {/* Column B: Instant Manual Handwriting Input */}
              <form
                onSubmit={handleManualSubmit}
                className="flex items-center gap-2 w-full"
              >
                <input
                  type="text"
                  value={manualText}
                  onChange={(e) => setManualText(e.target.value)}
                  placeholder="輸入中／日／韓／英文字，直接手寫上霧..."
                  className="w-full px-3 py-1.5 text-xs bg-[#0a0c11] border border-white/14 rounded-lg text-[#f3f6fa] placeholder:text-white/40 focus:outline-none focus:border-[#c8dcf0]/60"
                />
                <button
                  type="submit"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#1e222b] hover:bg-[#282d39] text-[#edf5ff] border border-white/14 transition-colors cursor-pointer whitespace-nowrap shrink-0"
                >
                  <PenTool className="w-3.5 h-3.5" />
                  <span>手寫上鏡</span>
                </button>
              </form>
            </div>

            {/* Expandable Scripted Queue OR Plain-Lyrics Alignment Drawer */}
            {showScriptDrawer && (
              <div className="p-2.5 rounded-lg bg-[#0b0d12] border border-white/10 space-y-2">
                <textarea
                  rows={3}
                  value={scriptText}
                  onChange={(e) => setScriptText(e.target.value)}
                  placeholder="貼上多行歌詞或台詞（一行一句）：可「自動輪流手寫」，或若已上傳音樂可直接「對齊音樂拍點」！"
                  className="w-full px-3 py-2 text-xs bg-[#12151c] border border-white/12 rounded-md text-[#f3f6fa] placeholder:text-white/40 focus:outline-none focus:border-[#c8dcf0]/60 resize-none leading-relaxed"
                />
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[11px] text-white/50">
                    大字寬距排版 · 自動保留留白不重疊
                  </span>
                  <div className="flex flex-wrap items-center gap-2">
                    {music.hasLoadedFile && (
                      <button
                        type="button"
                        disabled={music.isAnalyzing || !scriptText.trim()}
                        onClick={() => music.onAlignPlainLyrics(scriptText)}
                        className="px-3 py-1.5 text-xs font-medium rounded-md bg-[#c8dcf0]/18 hover:bg-[#c8dcf0]/28 border border-[#c8dcf0]/45 text-[#edf5ff] disabled:opacity-45 cursor-pointer transition-colors"
                      >
                        對齊已上傳音樂拍點
                      </button>
                    )}
                    {music.isScriptPlaying ? (
                      <button
                        type="button"
                        onClick={music.onStopTextQueue}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-[#b4362c]/35 border border-[#e87c72]/60 text-[#ffe1dd] cursor-pointer"
                      >
                        <Square className="w-3 h-3 fill-current" />
                        <span>停止輪寫</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        disabled={!scriptText.trim()}
                        onClick={() => music.onStartTextQueue(scriptText)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-[#e6eef7] hover:bg-white text-[#0d1016] disabled:opacity-40 cursor-pointer transition-colors"
                      >
                        <Play className="w-3 h-3 fill-current" />
                        <span>開始自動輪流手寫</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Playback Bar + Full Unwrapped Diagnostics & Lyrics Timeline */}
            {(music.sourceKind !== 'none' ||
              music.lyrics.length > 0 ||
              music.currentLyricText ||
              music.statusDetail ||
              music.errorMsg) && (
              <div className="space-y-2.5 pt-2.5 border-t border-white/8 text-xs">
                {music.sourceKind === 'file' && (
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={music.onTogglePlayPause}
                      aria-label={music.isPlaying ? '暫停音樂' : '播放音樂'}
                      className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-[#1e222b] hover:bg-[#282d39] text-[#edf5ff] border border-white/12 cursor-pointer shrink-0"
                    >
                      {music.isPlaying ? (
                        <Pause className="w-3.5 h-3.5" />
                      ) : (
                        <Play className="w-3.5 h-3.5" />
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={music.onToggleLoop}
                      aria-label="切換單曲循環"
                      className={`inline-flex items-center justify-center w-7 h-7 rounded-md border cursor-pointer shrink-0 transition-colors ${
                        music.loopTrack
                          ? 'bg-[#c8dcf0]/18 border-[#c8dcf0]/45 text-[#edf5ff]'
                          : 'bg-[#1e222b] border-white/12 text-white/55 hover:text-white'
                      }`}
                    >
                      <Repeat className="w-3.5 h-3.5" />
                    </button>

                    <span className="text-white/90 font-medium truncate max-w-[140px]">
                      {music.songTitle}
                    </span>

                    {(music.bpm || music.mood) && (
                      <span className="hidden sm:inline text-white/50 whitespace-nowrap shrink-0">
                        {[
                          music.bpm ? `${Math.round(music.bpm)} BPM` : null,
                          music.mood,
                        ]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                    )}

                    <input
                      type="range"
                      min={0}
                      max={Math.max(1, music.duration)}
                      step={0.05}
                      value={music.currentTime}
                      onChange={(e) => music.onSeek(parseFloat(e.target.value))}
                      className="mirror-slider flex-1 min-w-[85px]"
                    />

                    <span className="tabular-nums text-white/65 shrink-0">
                      {formatTime(music.currentTime)} /{' '}
                      {formatTime(music.duration)}
                    </span>

                    {music.lyrics.length > 0 && (
                      <div className="flex items-center gap-1 pl-2 border-l border-white/10 shrink-0">
                        <span className="text-white/50">微調對拍</span>
                        <button
                          type="button"
                          onClick={() => music.onAdjustOffset(-0.2)}
                          aria-label="歌詞提前 0.2 秒"
                          className="p-1 rounded bg-[#1a1e26] hover:bg-[#252b36] text-white/75 cursor-pointer"
                        >
                          <RotateCcw className="w-3 h-3" />
                        </button>
                        <span className="tabular-nums text-[#c8dcf0] w-11 text-center">
                          {music.lyricOffsetSec >= 0 ? '+' : ''}
                          {music.lyricOffsetSec.toFixed(1)}s
                        </span>
                        <button
                          type="button"
                          onClick={() => music.onAdjustOffset(0.2)}
                          aria-label="歌詞延後 0.2 秒"
                          className="p-1 rounded bg-[#1a1e26] hover:bg-[#252b36] text-white/75 cursor-pointer"
                        >
                          <RotateCw className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {music.lyrics.length > 0 && (
                  <div className="flex items-center gap-2 overflow-x-auto pb-1">
                    {music.lyrics.map((line, idx) => {
                      const isActive = idx === music.activeLineIndex;
                      return (
                        <button
                          key={`${line.startTimeSec}-${idx}`}
                          type="button"
                          onClick={() => music.onJumpToLyric(idx)}
                          className={`px-3 py-1.5 text-xs rounded-md border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                            isActive
                              ? 'bg-[#c8dcf0]/22 border-[#c8dcf0]/55 text-[#f4f8ff] font-medium'
                              : 'bg-[#0c0e13] border-white/10 text-white/65 hover:text-white hover:border-white/25'
                          }`}
                        >
                          <span className="tabular-nums text-[11px] text-white/40 mr-1.5">
                            {formatTime(line.startTimeSec)}
                          </span>
                          <span>{line.text}</span>
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Full, never-truncated Error Box OR Clear Live Diagnostic Status */}
                {music.errorMsg && (
                  <div className="flex flex-wrap items-start justify-between gap-2 bg-[#b4362c]/18 border border-[#e07064]/40 rounded-lg px-3 py-2 text-[#ffd2cc]">
                    <p className="flex-1 min-w-[200px] whitespace-normal break-words leading-relaxed">
                      {music.errorMsg}
                    </p>
                    {music.hasLoadedFile && !music.isAnalyzing && (
                      <button
                        type="button"
                        onClick={music.onRetryAnalysis}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded bg-[#e07064]/25 hover:bg-[#e07064]/40 border border-[#e07064]/50 text-white cursor-pointer shrink-0"
                      >
                        <RefreshCw className="w-3 h-3" />
                        <span>重新偵測歌詞</span>
                      </button>
                    )}
                  </div>
                )}

                {!music.errorMsg && (music.statusDetail || music.currentLyricText || music.isMicListening) && (
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    {music.currentLyricText ? (
                      <div className="flex items-start gap-1.5 min-w-0">
                        <span className="text-white/50 shrink-0">最新寫入鏡面：</span>
                        <span className="font-medium text-[#edf5ff] whitespace-normal break-words">
                          「{music.currentLyricText}」
                        </span>
                      </div>
                    ) : music.statusDetail ? (
                      <span className="text-[#c8dcf0]/90 whitespace-normal break-words leading-relaxed">
                        {music.statusDetail}
                      </span>
                    ) : music.isMicListening ? (
                      <span className="text-white/70 whitespace-normal leading-relaxed">
                        麥克風持續聽歌中（支援韓／日／繁中／英），聽到人聲將立即手寫上鏡...
                      </span>
                    ) : null}

                    {music.hasLoadedFile &&
                      !music.isAnalyzing &&
                      music.lyrics.length === 0 && (
                        <button
                          type="button"
                          onClick={music.onRetryAnalysis}
                          className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded bg-[#1c2029] hover:bg-[#252a35] border border-white/15 text-[#c8dcf0] cursor-pointer shrink-0"
                        >
                          <RefreshCw className="w-3 h-3" />
                          <span>重新偵測</span>
                        </button>
                      )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Spacious Grid for Continuous Fog Physics, Droplets, Stroke & Typography */}
        {activeTab === 'fog' && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-[#c8dcf0]/14 border border-[#c8dcf0]/30 text-[#e4f0fd]">
                  物理持續起霧模式（擦拭／寫字後自然慢慢回霧）
                </span>
                <button
                  type="button"
                  onClick={() =>
                    onChangeSettings({ breathCycle: !settings.breathCycle })
                  }
                  className={`px-2.5 py-1 text-xs rounded-md border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                    settings.breathCycle
                      ? 'bg-[#c8dcf0]/18 border-[#c8dcf0]/45 text-[#edf5ff]'
                      : 'bg-[#1a1e26] border-white/10 text-white/60 hover:text-white'
                  }`}
                >
                  {settings.breathCycle ? '微呼吸起伏：開啟' : '微呼吸起伏：關閉'}
                </button>
                <button
                  type="button"
                  onClick={() =>
                    onChangeSettings({ beatReactive: !settings.beatReactive })
                  }
                  className={`px-2.5 py-1 text-xs rounded-md border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                    settings.beatReactive
                      ? 'bg-[#c8dcf0]/18 border-[#c8dcf0]/45 text-[#edf5ff]'
                      : 'bg-[#1a1e26] border-white/10 text-white/60 hover:text-white'
                  }`}
                >
                  {settings.beatReactive ? '節拍水霧脈動：開啟' : '節拍水霧脈動：關閉'}
                </button>
              </div>

              <div className="flex items-center gap-1 p-0.5 bg-[#0c0e13] border border-white/10 rounded-md">
                {COVERAGE_PRESETS.map((preset) => {
                  const active =
                    Math.abs(settings.fogCoverage - preset.value) < 0.08;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() =>
                        onChangeSettings({ fogCoverage: preset.value })
                      }
                      className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                        active
                          ? 'bg-white/15 text-[#edf5ff]'
                          : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-x-4 gap-y-2.5 pt-1">
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="fog-stroke-weight" className="text-white/75">
                    筆畫粗細
                  </label>
                  <span className="text-[#c8dcf0] tabular-nums">
                    {settings.strokeWeight <= 0.28
                      ? '極細指尖'
                      : settings.strokeWeight <= 0.62
                        ? '細指'
                        : settings.strokeWeight <= 1.08
                          ? '標準'
                          : '粗抹'}
                  </span>
                </div>
                <input
                  id="fog-stroke-weight"
                  type="range"
                  min={0.15}
                  max={1.6}
                  step={0.05}
                  value={settings.strokeWeight}
                  onChange={(e) =>
                    onChangeSettings({
                      strokeWeight: parseFloat(e.target.value),
                    })
                  }
                  className="mirror-slider w-full"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <label
                    htmlFor="handwriting-scale-range"
                    className="text-white/75"
                  >
                    字體大小
                  </label>
                  <span className="text-[#c8dcf0] tabular-nums">
                    {settings.handwritingScale.toFixed(2)}x
                  </span>
                </div>
                <input
                  id="handwriting-scale-range"
                  type="range"
                  min={0.75}
                  max={1.4}
                  step={0.05}
                  value={settings.handwritingScale}
                  onChange={(e) =>
                    onChangeSettings({
                      handwritingScale: parseFloat(e.target.value),
                    })
                  }
                  className="mirror-slider w-full"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="fog-speed-range" className="text-white/75">
                    回霧速度
                  </label>
                  <span className="text-[#c8dcf0] tabular-nums">
                    {settings.recondenseSpeed.toFixed(1)}x
                  </span>
                </div>
                <input
                  id="fog-speed-range"
                  type="range"
                  min={0.6}
                  max={3.2}
                  step={0.2}
                  value={settings.recondenseSpeed}
                  onChange={(e) =>
                    onChangeSettings({
                      recondenseSpeed: parseFloat(e.target.value),
                    })
                  }
                  className="mirror-slider w-full"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="fog-temp-range" className="text-white/75">
                    溫差 ΔT
                  </label>
                  <span className="text-[#c8dcf0] tabular-nums">
                    +{deltaTempC}°C
                  </span>
                </div>
                <input
                  id="fog-temp-range"
                  type="range"
                  min={0.2}
                  max={1.0}
                  step={0.05}
                  value={settings.tempDiff}
                  onChange={(e) =>
                    onChangeSettings({ tempDiff: parseFloat(e.target.value) })
                  }
                  className="mirror-slider w-full"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="fog-density-range" className="text-white/75">
                    霧氣密度
                  </label>
                  <span className="text-[#c8dcf0] tabular-nums">
                    {densityPercent}%
                  </span>
                </div>
                <input
                  id="fog-density-range"
                  type="range"
                  min={0.25}
                  max={1.0}
                  step={0.05}
                  value={settings.maxDensity}
                  onChange={(e) =>
                    onChangeSettings({ maxDensity: parseFloat(e.target.value) })
                  }
                  className="mirror-slider w-full"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="fog-coverage-range" className="text-white/75">
                    霧氣範圍
                  </label>
                  <span className="text-[#c8dcf0] tabular-nums">
                    {coveragePercent}%
                  </span>
                </div>
                <input
                  id="fog-coverage-range"
                  type="range"
                  min={0.22}
                  max={1.0}
                  step={0.02}
                  value={settings.fogCoverage}
                  onChange={(e) =>
                    onChangeSettings({
                      fogCoverage: parseFloat(e.target.value),
                    })
                  }
                  className="mirror-slider w-full"
                />
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Video Editing Backdrop, Framing Guides & Export */}
        {activeTab === 'studio' && (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-white/55">鏡面底景：</span>
              <button
                type="button"
                onClick={() =>
                  onChangeSettings({
                    mirrorSource: 'studio',
                    studioTheme: 'daylight',
                  })
                }
                className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  settings.mirrorSource === 'studio' &&
                  settings.studioTheme === 'daylight'
                    ? 'bg-[#c8dcf0]/20 border-[#c8dcf0]/50 text-[#edf5ff]'
                    : 'bg-[#1c2029] border-white/10 text-white/70 hover:text-white'
                }`}
              >
                柔光石材空間（明亮日間）
              </button>

              <button
                type="button"
                onClick={() =>
                  onChangeSettings({
                    mirrorSource: 'studio',
                    studioTheme: 'velvet_slate',
                  })
                }
                className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  settings.mirrorSource === 'studio' &&
                  settings.studioTheme === 'velvet_slate'
                    ? 'bg-[#c8dcf0]/20 border-[#c8dcf0]/50 text-[#edf5ff]'
                    : 'bg-[#1c2029] border-white/10 text-white/70 hover:text-white'
                }`}
              >
                純淨柔霧深灰
              </button>

              <button
                type="button"
                onClick={() => onChangeSettings({ mirrorSource: 'camera' })}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  isCameraActive
                    ? 'bg-[#c8dcf0]/20 border-[#c8dcf0]/50 text-[#edf5ff]'
                    : 'bg-[#1c2029] border-white/10 text-white/70 hover:text-white'
                }`}
              >
                前置鏡頭實鏡
              </button>

              <input
                ref={mediaInputRef}
                type="file"
                accept="image/*,video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) recording.onUploadBackdropMedia(file);
                  e.target.value = '';
                }}
              />
              <button
                type="button"
                onClick={() => mediaInputRef.current?.click()}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  isMediaActive
                    ? 'bg-[#c8dcf0]/20 border-[#c8dcf0]/50 text-[#edf5ff]'
                    : 'bg-[#1c2029] border-white/10 text-white/75 hover:text-white'
                }`}
              >
                <ImagePlus className="w-3.5 h-3.5 text-[#c8dcf0]" />
                <span>
                  {recording.hasCustomMedia && recording.customMediaName
                    ? `素材：${recording.customMediaName}`
                    : '匯入剪片底圖／影片'}
                </span>
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-white/55">構圖參考線：</span>
              <div className="flex items-center gap-0.5 p-0.5 bg-[#0c0e13] border border-white/10 rounded-lg">
                {(
                  [
                    { id: 'none', label: '無' },
                    { id: '9:16', label: '9:16 直式短影音' },
                    { id: '4:5', label: '4:5 IG' },
                  ] as const
                ).map((g) => (
                  <button
                    key={g.id}
                    type="button"
                    onClick={() => recording.onChangeFramingGuide(g.id)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                      recording.framingGuide === g.id
                        ? 'bg-white/16 text-[#edf5ff]'
                        : 'text-white/60 hover:text-white'
                    }`}
                  >
                    {g.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

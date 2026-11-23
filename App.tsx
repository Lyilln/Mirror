import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  MirrorViewport,
  type MirrorViewportHandle,
} from './components/MirrorViewport';
import {
  MirrorControlsDock,
  type FramingGuide,
} from './components/MirrorControlsDock';
import { useMusicLyricsSync } from './hooks/useMusicLyricsSync';
import type { FogStatus, MirrorSettings } from './mirror/types';

// Phones: prefer hardware H.264 (mp4) and a lower bitrate; real-time VP9 at 10 Mbps
// competes with the shaders for the same GPU/CPU budget. Desktop keeps VP9.
function pickRecorderMime(): { mimeType: string; ext: 'mp4' | 'webm'; bitrate: number } {
  const isTouch =
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(pointer: coarse)').matches;
  const candidates = isTouch
    ? [
        'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
        'video/mp4;codecs=avc1,mp4a.40.2',
        'video/webm;codecs=h264,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm;codecs=vp9,opus',
        'video/webm',
        'video/mp4',
      ]
    : [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm;codecs=h264,opus',
        'video/webm',
        'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
        'video/mp4',
      ];
  const bitrate = isTouch ? 6_000_000 : 10_000_000;
  for (const mimeType of candidates) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType, ext: mimeType.startsWith('video/mp4') ? 'mp4' : 'webm', bitrate };
    }
  }
  return { mimeType: '', ext: 'webm', bitrate };
}

export default function App() {
  const viewportRef = useRef<MirrorViewportHandle>(null);

  const [settings, setSettings] = useState<MirrorSettings>({
    breathCycle: false,
    recondenseSpeed: 1.4,
    maxDensity: 0.92,
    fogCoverage: 1.0,
    tempDiff: 0.65,
    mirrorSource: 'studio',
    studioTheme: 'daylight',
    lyricsActive: false,
    handwritingScale: 1.05,
    strokeWeight: 0.36,
    beatReactive: true,
  });

  const [fogStatus, setFogStatus] = useState<FogStatus>({
    fogLevel: 0.92,
    phase: 'fogging',
    wipedRatio: 0,
  });

  const [dockCollapsed, setDockCollapsed] = useState(false);
  const [showPureHint, setShowPureHint] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [framingGuide, setFramingGuide] = useState<FramingGuide>('none');

  const [customMediaUrl, setCustomMediaUrl] = useState<string | null>(null);
  const [customMediaType, setCustomMediaType] = useState<
    'image' | 'video' | null
  >(null);
  const [customMediaName, setCustomMediaName] = useState<string | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [recordingSec, setRecordingSec] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const recordTimerRef = useRef<number>(0);

  const handleLyricDetected = useCallback(
    (text: string, durationSec?: number, isChorus?: boolean) => {
      viewportRef.current?.writeLyricOnGlass(text, durationSec, isChorus);
    },
    [],
  );

  const handleEnergyUpdate = useCallback((energy: number) => {
    viewportRef.current?.setAudioEnergy(energy);
  }, []);

  const handleBeatPulse = useCallback((intensity: number) => {
    viewportRef.current?.triggerBeatPulse(intensity);
  }, []);

  const musicSync = useMusicLyricsSync({
    onLyricDetected: handleLyricDetected,
    onEnergyUpdate: handleEnergyUpdate,
    onBeatPulse: handleBeatPulse,
  });

  const lyricsActive = musicSync.isPlaying || musicSync.isMicListening;

  // Derived value: no effect + extra render needed to mirror lyricsActive into settings
  const effectiveSettings = useMemo<MirrorSettings>(
    () =>
      settings.lyricsActive === lyricsActive
        ? settings
        : { ...settings, lyricsActive },
    [settings, lyricsActive],
  );

  // musicSync is a new object every render; read it through a ref so callbacks and
  // window listeners stay stable instead of being re-created / re-bound each render.
  const musicRef = useRef(musicSync);
  musicRef.current = musicSync;

  // Only re-render App (and the dock) when the status visibly changes
  const handleFogStatus = useCallback((next: FogStatus) => {
    setFogStatus((prev) =>
      prev.phase === next.phase &&
      Math.abs(prev.fogLevel - next.fogLevel) < 0.02 &&
      Math.abs(prev.wipedRatio - next.wipedRatio) < 0.02
        ? prev
        : next,
    );
  }, []);

  const handleSettingsChange = useCallback((patch: Partial<MirrorSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const handleCameraFallback = useCallback(() => {
    setSettings((prev) => ({ ...prev, mirrorSource: 'studio' }));
  }, []);

  const togglePureMode = useCallback(() => {
    setDockCollapsed((prev) => {
      const next = !prev;
      setShowPureHint(next);
      return next;
    });
  }, []);

  useEffect(() => {
    if (!showPureHint) return;
    const t = window.setTimeout(() => setShowPureHint(false), 2200);
    return () => window.clearTimeout(t);
  }, [showPureHint]);

  const handleUploadBackdropMedia = useCallback(
    (file: File) => {
      if (customMediaUrl) {
        URL.revokeObjectURL(customMediaUrl);
      }
      const url = URL.createObjectURL(file);
      const isVid = file.type.startsWith('video/');
      setCustomMediaUrl(url);
      setCustomMediaType(isVid ? 'video' : 'image');
      setCustomMediaName(file.name.replace(/\.[^.]+$/, ''));
      setSettings((prev) => ({ ...prev, mirrorSource: 'media' }));
    },
    [customMediaUrl],
  );

  useEffect(() => {
    return () => {
      if (customMediaUrl) {
        URL.revokeObjectURL(customMediaUrl);
      }
    };
  }, [customMediaUrl]);

  const handleToggleRecording = useCallback(() => {
    if (isRecording && recorderRef.current) {
      recorderRef.current.stop();
      return;
    }

    const canvas = viewportRef.current?.getCanvasElement();
    if (!canvas) return;

    try {
      const canvasStream = canvas.captureStream(60);
      const tracks: MediaStreamTrack[] = [...canvasStream.getVideoTracks()];

      const audioStream = musicRef.current.getRecordingAudioStream();
      if (audioStream) {
        audioStream.getAudioTracks().forEach((t) => tracks.push(t));
      }

      const combinedStream = new MediaStream(tracks);
      const picked = pickRecorderMime();

      const recorder = new MediaRecorder(combinedStream, {
        ...(picked.mimeType ? { mimeType: picked.mimeType } : {}),
        videoBitsPerSecond: picked.bitrate,
      });

      recordedChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        window.clearInterval(recordTimerRef.current);
        canvasStream.getVideoTracks().forEach((t) => t.stop());
        setIsRecording(false);
        setRecordingSec(0);
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || picked.mimeType || 'video/webm',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fog-mirror-edit-${Date.now()}.${picked.ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.setTimeout(() => URL.revokeObjectURL(url), 4000);
      };

      recorderRef.current = recorder;
      recorder.start(250);
      setIsRecording(true);
      setRecordingSec(0);
      recordTimerRef.current = window.setInterval(() => {
        setRecordingSec((s) => s + 1);
      }, 1000);
    } catch {
      setIsRecording(false);
    }
  }, [isRecording]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if (e.key === 'h' || e.key === 'H') {
        togglePureMode();
      } else if (e.key === ' ' && musicRef.current.sourceKind === 'file') {
        e.preventDefault();
        void musicRef.current.togglePlayPause();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [togglePureMode]);

  return (
    <main className="relative w-screen h-[100dvh] overflow-hidden bg-[#1a1c20] select-none cursor-crosshair">
      {/* 100% Full-Screen Frameless Frosted Mirror (Zero Overlay Title Text) */}
      <MirrorViewport
        ref={viewportRef}
        settings={effectiveSettings}
        customMediaUrl={customMediaUrl}
        customMediaType={customMediaType}
        onFogStatusChange={handleFogStatus}
        onCameraError={setCameraError}
        onCameraFallback={handleCameraFallback}
        onDoubleTapGlass={togglePureMode}
      />

      {/* Optional Non-Recorded Framing Safe-Area Guides for Reels / TikTok / IG */}
      {framingGuide !== 'none' && !dockCollapsed && (
        <div className="fixed inset-0 z-10 pointer-events-none flex items-center justify-center">
          <div
            className={`border border-white/30 border-dashed rounded-lg ${
              framingGuide === '9:16'
                ? 'h-[86vh] aspect-[9/16]'
                : 'h-[82vh] aspect-[4/5]'
            }`}
          />
        </div>
      )}

      {/* Transient 2.2s whisper only when hiding dock; fades completely to 0 opacity */}
      <div
        className={`fixed top-5 left-1/2 -translate-x-1/2 z-20 pointer-events-none transition-opacity duration-500 ${
          showPureHint ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <p className="px-3.5 py-1.5 text-xs text-white/85 bg-black/55 rounded-lg border border-white/10">
          純淨錄影模式 · 雙擊畫面恢復面板
        </p>
      </div>

      {/* Collapsible Unified Bottom Control Dock */}
      <MirrorControlsDock
        settings={effectiveSettings}
        fogStatus={fogStatus}
        cameraError={cameraError}
        collapsed={dockCollapsed}
        music={{
          sourceKind: musicSync.sourceKind,
          detectionStage: musicSync.detectionStage,
          statusDetail: musicSync.statusDetail,
          songTitle: musicSync.songTitle,
          bpm: musicSync.bpm,
          mood: musicSync.mood,
          isAnalyzing: musicSync.isAnalyzing,
          isPlaying: musicSync.isPlaying,
          isMicListening: musicSync.isMicListening,
          isScriptPlaying: musicSync.isScriptPlaying,
          hasLoadedFile: musicSync.hasLoadedFile,
          currentTime: musicSync.currentTime,
          duration: musicSync.duration,
          lyrics: musicSync.lyrics,
          activeLineIndex: musicSync.activeLineIndex,
          currentLyricText: musicSync.currentLyricText,
          hasVocals: musicSync.hasVocals,
          loopTrack: musicSync.loopTrack,
          lyricOffsetSec: musicSync.lyricOffsetSec,
          errorMsg: musicSync.errorMsg,
          onToggleLoop: () => musicSync.setLoopTrack((v) => !v),
          onAdjustOffset: (d) =>
            musicSync.setLyricOffsetSec((v) =>
              Math.max(-3, Math.min(3, Math.round((v + d) * 10) / 10)),
            ),
          onUploadFile: musicSync.loadAudioFile,
          onRetryAnalysis: musicSync.retryAnalysis,
          onAlignPlainLyrics: musicSync.alignPlainLyricsWithAudio,
          onImportLyricsFile: musicSync.importSyncedLyricsFile,
          onTogglePlayPause: musicSync.togglePlayPause,
          onSeek: musicSync.seekTo,
          onJumpToLyric: musicSync.jumpToLyricLine,
          onToggleMic: () =>
            musicSync.isMicListening
              ? musicSync.stopMic()
              : musicSync.startMicListening(),
          onWriteManualLine: musicSync.writeManualLine,
          onStartTextQueue: musicSync.startTextQueue,
          onStopTextQueue: musicSync.stopTextQueue,
        }}
        recording={{
          isRecording,
          recordingSec,
          framingGuide,
          hasCustomMedia: Boolean(customMediaUrl),
          customMediaName,
          onToggleRecording: handleToggleRecording,
          onChangeFramingGuide: setFramingGuide,
          onUploadBackdropMedia: handleUploadBackdropMedia,
        }}
        onToggleCollapse={togglePureMode}
        onChangeSettings={handleSettingsChange}
        onSteamBurst={() => viewportRef.current?.triggerSteamBurst()}
        onClearFog={() => viewportRef.current?.clearAllFog()}
      />
    </main>
  );
}

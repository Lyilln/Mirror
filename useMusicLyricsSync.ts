import { useCallback, useEffect, useRef, useState } from 'react';
import type { LyricLine } from '../mirror/types';

export type AudioSourceKind = 'none' | 'file' | 'mic';
export type DetectionStage =
  | 'idle'
  | 'reading'
  | 'analyzing'
  | 'ready'
  | 'no-vocals'
  | 'error';

interface UseMusicLyricsSyncOptions {
  onLyricDetected: (
    text: string,
    durationSec?: number,
    isChorus?: boolean,
  ) => void;
  onEnergyUpdate: (energy: number) => void;
  onBeatPulse: (intensity: number) => void;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

function normalizePhrase(str: string): string {
  return str
    .toLowerCase()
    .replace(/[\s,.!?，。！？、·'"`~—\-()（）[\]【】]/g, '');
}

// Decode -> mono -> low sample-rate 16-bit WAV, so big files fit under proxy upload limits (HTTP 413).
async function compressAudioForUpload(blob: Blob, targetRate: number): Promise<Blob> {
  const Ctor: typeof AudioContext | undefined =
    window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) throw new Error('此瀏覽器無法壓縮音訊。');
  const ctx = new Ctor();
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    const frames = Math.max(1, Math.ceil(decoded.duration * targetRate));
    const off = new OfflineAudioContext(1, frames, targetRate);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    const pcm = rendered.getChannelData(0);
    const buf = new ArrayBuffer(44 + pcm.length * 2);
    const dv = new DataView(buf);
    const wr = (o: number, t: string) => { for (let i = 0; i < t.length; i++) dv.setUint8(o + i, t.charCodeAt(i)); };
    wr(0, 'RIFF'); dv.setUint32(4, 36 + pcm.length * 2, true); wr(8, 'WAVE'); wr(12, 'fmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, targetRate, true); dv.setUint32(28, targetRate * 2, true);
    dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); wr(36, 'data'); dv.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) {
      const v = Math.max(-1, Math.min(1, pcm[i]));
      dv.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  } finally {
    void ctx.close();
  }
}

function formatFriendlyError(err: unknown, context: 'mic' | 'file' | 'play'): string {
  const raw = err instanceof Error ? err.message : String(err || '');
  const isNotAllowed =
    (err instanceof DOMException && err.name === 'NotAllowedError') ||
    /not allowed by the user agent|not allowed by the platform|NotAllowed|permission denied/i.test(raw);

  if (isNotAllowed) {
    if (context === 'mic') {
      return '此預覽視窗或瀏覽器未開放麥克風權限（NotAllowedError）。請改用「上傳音樂偵測歌詞」或匯入 .lrc／直接手寫，或在新分頁獨立開啟網頁後再啟用麥克風。';
    }
    if (context === 'play') {
      return 'iOS／瀏覽器限制自動播放音訊，請手動點擊一次「▶ 播放」按鈕即可開始對拍與手寫歌詞。';
    }
    return '瀏覽器或裝置暫時不允許讀取此音訊來源（NotAllowedError），請改從「檔案」App 選擇本地音訊檔再試一次。';
  }

  return raw || '偵測過程發生未預期錯誤。';
}

function parseTimestamp(str: string): number | null {
  const srt = str.match(/(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})/);
  if (srt) {
    const h = Number(srt[1]);
    const m = Number(srt[2]);
    const s = Number(srt[3]);
    const ms = Number(srt[4].padEnd(3, '0'));
    return h * 3600 + m * 60 + s + ms / 1000;
  }
  const lrc = str.match(/(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?/);
  if (lrc) {
    const m = Number(lrc[1]);
    const s = Number(lrc[2]);
    const frac = lrc[3] ? Number(lrc[3].padEnd(3, '0')) / 1000 : 0;
    return m * 60 + s + frac;
  }
  return null;
}

function parseLrcOrSrt(rawText: string): LyricLine[] {
  const text = rawText.replace(/\r\n/g, '\n');

  if (/-->/.test(text)) {
    const blocks = text.split(/\n\s*\n/);
    const lines: LyricLine[] = [];
    for (const blk of blocks) {
      const rows = blk.split('\n').map((r) => r.trim()).filter(Boolean);
      const arrowIdx = rows.findIndex((r) => r.includes('-->'));
      if (arrowIdx === -1) continue;
      const [startStr, endStr] = rows[arrowIdx].split('-->');
      const start = parseTimestamp(startStr);
      const end = parseTimestamp(endStr);
      const content = rows.slice(arrowIdx + 1).join(' ').trim();
      if (start !== null && content) {
        lines.push({
          startTimeSec: start,
          endTimeSec: end !== null && end > start ? end : start + 2.5,
          text: content,
        });
      }
    }
    return lines.sort((a, b) => a.startTimeSec - b.startTimeSec);
  }

  const lrcItems: { start: number; text: string }[] = [];
  for (const row of text.split('\n')) {
    const tagMatches = [...row.matchAll(/\[(\d{1,3}:\d{2}(?:\.\d{1,3})?)\]/g)];
    if (tagMatches.length === 0) continue;
    const lyric = row.replace(/\[[^\]]+\]/g, '').trim();
    if (!lyric) continue;
    for (const m of tagMatches) {
      const sec = parseTimestamp(m[1]);
      if (sec !== null) {
        lrcItems.push({ start: sec, text: lyric });
      }
    }
  }
  lrcItems.sort((a, b) => a.start - b.start);
  return lrcItems.map((item, idx) => {
    const nextStart = lrcItems[idx + 1]?.start;
    const end = nextStart ? Math.min(nextStart - 0.05, item.start + 4.5) : item.start + 3.0;
    return {
      startTimeSec: item.start,
      endTimeSec: Math.max(item.start + 0.85, end),
      text: item.text,
    };
  });
}

export function useMusicLyricsSync({
  onLyricDetected,
  onEnergyUpdate,
  onBeatPulse,
}: UseMusicLyricsSyncOptions) {
  const [sourceKind, setSourceKind] = useState<AudioSourceKind>('none');
  const [detectionStage, setDetectionStage] = useState<DetectionStage>('idle');
  const [statusDetail, setStatusDetail] = useState<string>('');
  const [songTitle, setSongTitle] = useState<string>('');
  const [bpm, setBpm] = useState<number | null>(null);
  const [mood, setMood] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMicListening, setIsMicListening] = useState(false);
  const [isScriptPlaying, setIsScriptPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [lyrics, setLyrics] = useState<LyricLine[]>([]);
  const [activeLineIndex, setActiveLineIndex] = useState<number>(-1);
  const [currentLyricText, setCurrentLyricText] = useState<string>('');
  const [hasVocals, setHasVocals] = useState<boolean | null>(null);
  const [loopTrack, setLoopTrack] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [lyricOffsetSec, setLyricOffsetSec] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const stableBlobRef = useRef<{ blob: Blob; fileName: string } | null>(null);
  const mediaElRef = useRef<HTMLVideoElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const recordDestRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micRecorderActiveRef = useRef(false);
  const scriptTimerRef = useRef<number>(0);
  const lastTriggeredIndexRef = useRef<number>(-1);
  const lastAudioTimeRef = useRef<number>(0);
  const lastMicPhraseRef = useRef<string>('');
  const lyricsRef = useRef<LyricLine[]>([]);
  const lyricOffsetRef = useRef<number>(0);
  const rafRef = useRef<number>(0);

  const onLyricRef = useRef(onLyricDetected);
  const onEnergyRef = useRef(onEnergyUpdate);
  const onBeatRef = useRef(onBeatPulse);
  onLyricRef.current = onLyricDetected;
  onEnergyRef.current = onEnergyUpdate;
  onBeatRef.current = onBeatPulse;
  lyricsRef.current = lyrics;
  lyricOffsetRef.current = lyricOffsetSec;

  const stopTextQueue = useCallback(() => {
    window.clearInterval(scriptTimerRef.current);
    scriptTimerRef.current = 0;
    setIsScriptPlaying(false);
  }, []);

  const ensureAudioGraph = useCallback(async (mediaEl: HTMLVideoElement) => {
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      if (audioCtxRef.current.state === 'suspended') {
        await audioCtxRef.current.resume();
      }
      if (!mediaSourceRef.current) {
        mediaSourceRef.current =
          audioCtxRef.current.createMediaElementSource(mediaEl);
        analyserRef.current = audioCtxRef.current.createAnalyser();
        analyserRef.current.fftSize = 512;
        analyserRef.current.smoothingTimeConstant = 0.72;
        recordDestRef.current = audioCtxRef.current.createMediaStreamDestination();

        mediaSourceRef.current.connect(analyserRef.current);
        analyserRef.current.connect(audioCtxRef.current.destination);
        analyserRef.current.connect(recordDestRef.current);
      }
      return analyserRef.current;
    } catch {
      return null;
    }
  }, []);

  const stopSyncLoop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    onEnergyRef.current(0);
  }, []);

  // Runs both WebAudio beat/energy telemetry (if available) AND lyric timeline sync independently!
  const startSyncLoop = useCallback(
    (analyser: AnalyserNode | null, syncWithMediaEl: boolean) => {
      cancelAnimationFrame(rafRef.current);
      const data = analyser
        ? new Uint8Array(analyser.frequencyBinCount)
        : null;
      let prevBass = 0;
      let lastBeatMs = 0;
      let lastUiTimeMs = 0;

      const loop = (now: number) => {
        if (analyser && data) {
          analyser.getByteFrequencyData(data);

          let bassSum = 0;
          const bassBins = Math.min(12, data.length);
          for (let i = 0; i < bassBins; i++) {
            bassSum += data[i];
          }
          const bassNorm = Math.min(1, bassSum / (bassBins * 195));

          let midSum = 0;
          const midEnd = Math.min(54, data.length);
          for (let i = bassBins; i < midEnd; i++) {
            midSum += data[i];
          }
          const midNorm = Math.min(1, midSum / Math.max(1, (midEnd - bassBins) * 165));

          const combinedEnergy = Math.min(1, bassNorm * 0.65 + midNorm * 0.45);
          onEnergyRef.current(combinedEnergy);

          if (bassNorm - prevBass > 0.16 && bassNorm > 0.42 && now - lastBeatMs > 230) {
            lastBeatMs = now;
            onBeatRef.current(Math.min(1, bassNorm * 1.15));
          }
          prevBass = prevBass * 0.78 + bassNorm * 0.22;
        }

        if (syncWithMediaEl && mediaElRef.current) {
          const t = mediaElRef.current.currentTime;

          if (t + 0.5 < lastAudioTimeRef.current) {
            lastTriggeredIndexRef.current = -1;
          }
          lastAudioTimeRef.current = t;

          if (now - lastUiTimeMs > 85) {
            lastUiTimeMs = now;
            setCurrentTime(t);
          }

          const linesList = lyricsRef.current;
          const shiftedT = t - lyricOffsetRef.current;
          if (linesList.length > 0) {
            let activeIdx = -1;
            for (let i = 0; i < linesList.length; i++) {
              if (
                shiftedT >= linesList[i].startTimeSec &&
                shiftedT <= linesList[i].endTimeSec + 0.45
              ) {
                activeIdx = i;
              }
            }
            if (activeIdx !== -1 && activeIdx !== lastTriggeredIndexRef.current) {
              lastTriggeredIndexRef.current = activeIdx;
              const line = linesList[activeIdx];
              setActiveLineIndex(activeIdx);
              setCurrentLyricText(line.text);
              const writeDur = Math.min(
                0.82,
                Math.max(0.34, (line.endTimeSec - line.startTimeSec) * 0.42),
              );
              onLyricRef.current(line.text, writeDur, line.isChorus);
            }
          }
        }

        rafRef.current = requestAnimationFrame(loop);
      };

      rafRef.current = requestAnimationFrame(loop);
    },
    [],
  );

  const stopMic = useCallback(() => {
    micRecorderActiveRef.current = false;
    setIsMicListening(false);
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    stopSyncLoop();
  }, [stopSyncLoop]);

  const stopAudioFile = useCallback(() => {
    if (mediaElRef.current) {
      mediaElRef.current.pause();
      setIsPlaying(false);
    }
    stopSyncLoop();
  }, [stopSyncLoop]);

  useEffect(() => {
    if (mediaElRef.current) {
      mediaElRef.current.loop = loopTrack;
    }
  }, [loopTrack]);

  useEffect(() => {
    if (mediaElRef.current) {
      mediaElRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  const analyzeBlobWithServer = useCallback(
    async (blob: Blob, fileName: string, plainLyrics?: string) => {
      setIsAnalyzing(true);
      setErrorMsg(null);
      setDetectionStage('analyzing');
      const mb = (blob.size / (1024 * 1024)).toFixed(1);
      setStatusDetail(`AI 正在聆聽音訊（${mb} MB）並精準對齊人聲歌詞時間軸...`);

      try {
        const post = (b: Blob, name: string) => {
          const fd = new FormData();
          fd.append('audio', b, name);
          if (plainLyrics && plainLyrics.trim()) fd.append('plainLyrics', plainLyrics.trim());
          return fetch('/api/analyze-lyrics', { method: 'POST', body: fd });
        };
        let resp = await post(blob, fileName);
        // HTTP 413 (proxy / platform body limit): shrink the audio and retry automatically.
        for (const rate of [16000, 11025, 8000]) {
          if (resp.status !== 413) break;
          setStatusDetail(`檔案超過上傳上限，正在壓縮音訊（${rate / 1000} kHz 單聲道）後重試...`);
          const small = await compressAudioForUpload(blob, rate);
          resp = await post(small, fileName.replace(/\.[^.]+$/, '') + '.wav');
        }
        if (resp.status === 413) {
          throw new Error('音訊檔太大，即使壓縮後仍超過伺服器上限。請改傳較短的片段（建議 5 分鐘內），或直接貼上歌詞／匯入 .lrc。');
        }

        let data: {
          error?: string;
          songTitle?: string;
          hasVocals?: boolean;
          bpm?: number;
          mood?: string;
          lines?: LyricLine[];
        } = {};
        try {
          data = await resp.json();
        } catch {
          throw new Error(`伺服器回應異常（HTTP ${resp.status}），請檢查音訊格式或伺服器狀態。`);
        }

        if (!resp.ok) {
          throw new Error(data.error || `歌詞解析失敗（HTTP ${resp.status}）`);
        }

        const detectedLines = data.lines ?? [];
        setSongTitle(data.songTitle || fileName.replace(/\.[^.]+$/, ''));
        setBpm(data.bpm ?? null);
        setMood(data.mood ?? null);
        const vocals = Boolean(data.hasVocals && detectedLines.length > 0);
        setHasVocals(vocals);
        setLyrics(detectedLines);
        lyricsRef.current = detectedLines;
        lastTriggeredIndexRef.current = -1;
        setActiveLineIndex(-1);

        if (vocals) {
          setDetectionStage('ready');
          setStatusDetail(`偵測成功：已辨識 ${detectedLines.length} 句歌詞時間軸`);
        } else {
          setDetectionStage('no-vocals');
          setStatusDetail('AI 分析完成：未偵測到人聲歌詞（可能為純音樂），您也可貼上歌詞對齊或直接手動寫字');
        }
      } catch (err) {
        setDetectionStage('error');
        setStatusDetail('');
        setErrorMsg(formatFriendlyError(err, 'file'));
      } finally {
        setIsAnalyzing(false);
      }
    },
    [],
  );

  // Immediately reads file bytes into a stable memory Blob so iOS Safari never revokes file permissions mid-upload!
  const loadAudioFile = useCallback(
    async (file: File) => {
      stopMic();
      stopTextQueue();
      stopAudioFile();
      setErrorMsg(null);
      setDetectionStage('reading');
      setStatusDetail(`正在讀取音訊檔（${file.name}）...`);
      setHasVocals(null);
      setLyrics([]);
      setActiveLineIndex(-1);
      setCurrentLyricText('');
      setBpm(null);
      setMood(null);
      lastTriggeredIndexRef.current = -1;
      lastAudioTimeRef.current = 0;
      setSourceKind('file');
      setSongTitle(file.name.replace(/\.[^.]+$/, ''));

      let stableBlob: Blob;
      try {
        const buffer = await file.arrayBuffer();
        stableBlob = new Blob([buffer], {
          type: file.type || 'audio/mpeg',
        });
        stableBlobRef.current = { blob: stableBlob, fileName: file.name };
      } catch (err) {
        setDetectionStage('error');
        setStatusDetail('');
        setErrorMsg(formatFriendlyError(err, 'file'));
        return;
      }

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const url = URL.createObjectURL(stableBlob);
      objectUrlRef.current = url;

      if (!mediaElRef.current) {
        const vid = document.createElement('video');
        vid.playsInline = true;
        vid.preload = 'auto';
        mediaElRef.current = vid;
      }
      const mediaEl = mediaElRef.current;
      mediaEl.src = url;
      mediaEl.loop = loopTrack;
      mediaEl.playbackRate = playbackRate;
      mediaEl.load();

      mediaEl.onloadedmetadata = () => {
        setDuration(mediaEl.duration || 0);
      };
      mediaEl.onended = () => {
        if (!mediaEl.loop) {
          setIsPlaying(false);
          stopSyncLoop();
        }
      };

      await analyzeBlobWithServer(stableBlob, file.name);

      try {
        const analyser = await ensureAudioGraph(mediaEl);
        await mediaEl.play();
        setIsPlaying(true);
        startSyncLoop(analyser, true);
      } catch {
        // Autoplay after async network fetch is often restricted on iOS Safari —
        // keep status clean and let user press Play manually without showing a scary red error!
      }
    },
    [
      analyzeBlobWithServer,
      ensureAudioGraph,
      loopTrack,
      playbackRate,
      startSyncLoop,
      stopAudioFile,
      stopMic,
      stopSyncLoop,
      stopTextQueue,
    ],
  );

  const retryAnalysis = useCallback(async () => {
    if (!stableBlobRef.current) return;
    await analyzeBlobWithServer(
      stableBlobRef.current.blob,
      stableBlobRef.current.fileName,
    );
  }, [analyzeBlobWithServer]);

  const alignPlainLyricsWithAudio = useCallback(
    async (plainText: string) => {
      const clean = plainText.trim();
      if (!clean) return;
      if (!stableBlobRef.current) {
        setErrorMsg('請先上傳音樂檔案，即可將歌詞精準對齊音樂拍點。');
        return;
      }
      await analyzeBlobWithServer(
        stableBlobRef.current.blob,
        stableBlobRef.current.fileName,
        clean,
      );
    },
    [analyzeBlobWithServer],
  );

  const importSyncedLyricsFile = useCallback(async (file: File) => {
    try {
      const content = await file.text();
      const parsed = parseLrcOrSrt(content);
      if (parsed.length === 0) {
        setDetectionStage('error');
        setErrorMsg('無法解析歌詞時間軸，請確認為標準 .lrc 或 .srt 格式。');
        return;
      }
      setErrorMsg(null);
      setDetectionStage('ready');
      setStatusDetail(`已匯入 ${parsed.length} 句時間軸歌詞`);
      setLyrics(parsed);
      lyricsRef.current = parsed;
      setHasVocals(true);
      lastTriggeredIndexRef.current = -1;
      setActiveLineIndex(-1);
    } catch (err) {
      setDetectionStage('error');
      setErrorMsg(formatFriendlyError(err, 'file'));
    }
  }, []);

  const togglePlayPause = useCallback(async () => {
    const mediaEl = mediaElRef.current;
    if (!mediaEl || sourceKind !== 'file') return;

    if (mediaEl.paused) {
      try {
        const analyser = await ensureAudioGraph(mediaEl);
        await mediaEl.play();
        setIsPlaying(true);
        setErrorMsg(null);
        startSyncLoop(analyser, true);
      } catch (err) {
        setErrorMsg(formatFriendlyError(err, 'play'));
      }
    } else {
      mediaEl.pause();
      setIsPlaying(false);
      stopSyncLoop();
    }
  }, [ensureAudioGraph, sourceKind, startSyncLoop, stopSyncLoop]);

  const seekTo = useCallback((sec: number) => {
    if (!mediaElRef.current) return;
    mediaElRef.current.currentTime = sec;
    lastAudioTimeRef.current = sec;
    setCurrentTime(sec);
    lastTriggeredIndexRef.current = -1;
  }, []);

  const jumpToLyricLine = useCallback((idx: number) => {
    const line = lyricsRef.current[idx];
    if (!line) return;
    setActiveLineIndex(idx);
    setCurrentLyricText(line.text);
    onLyricRef.current(line.text, 0.54, line.isChorus);
    if (mediaElRef.current) {
      const target = Math.max(0, line.startTimeSec + lyricOffsetRef.current);
      mediaElRef.current.currentTime = target;
      lastAudioTimeRef.current = target;
      setCurrentTime(target);
      lastTriggeredIndexRef.current = idx;
    }
  }, []);

  const startMicListening = useCallback(async () => {
    stopAudioFile();
    stopTextQueue();
    setErrorMsg(null);
    setDetectionStage('idle');
    setStatusDetail('');
    setLyrics([]);
    setActiveLineIndex(-1);
    setHasVocals(null);
    setBpm(null);
    setMood(null);
    setCurrentLyricText('');
    lastMicPhraseRef.current = '';

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('目前瀏覽器環境不支援麥克風錄音介面（getUserMedia）。');
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;
      setSourceKind('mic');
      setSongTitle('麥克風即時聽歌');
      setIsMicListening(true);
      micRecorderActiveRef.current = true;

      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      if (audioCtxRef.current.state === 'suspended') {
        await audioCtxRef.current.resume();
      }
      const micSource = audioCtxRef.current.createMediaStreamSource(stream);
      const micAnalyser = audioCtxRef.current.createAnalyser();
      micAnalyser.fftSize = 512;
      micAnalyser.smoothingTimeConstant = 0.72;
      micSource.connect(micAnalyser);
      startSyncLoop(micAnalyser, false);

      const timeData = new Uint8Array(micAnalyser.fftSize);
      let slicePeakLevel = 0;
      let sliceRmsAccum = 0;
      let sliceSamples = 0;
      const sampleTimer = window.setInterval(() => {
        if (!micRecorderActiveRef.current) {
          window.clearInterval(sampleTimer);
          return;
        }
        micAnalyser.getByteTimeDomainData(timeData);
        let peak = 0;
        let sqSum = 0;
        for (let i = 0; i < timeData.length; i++) {
          const v = (timeData[i] - 128) / 128;
          const absV = Math.abs(v);
          if (absV > peak) peak = absV;
          sqSum += v * v;
        }
        const rms = Math.sqrt(sqSum / timeData.length);
        if (peak > slicePeakLevel) slicePeakLevel = peak;
        sliceRmsAccum += rms;
        sliceSamples += 1;
      }, 100);

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : 'audio/mp4';

      const recordNextSlice = () => {
        if (!micRecorderActiveRef.current || !micStreamRef.current) return;

        slicePeakLevel = 0;
        sliceRmsAccum = 0;
        sliceSamples = 0;
        const chunks: BlobPart[] = [];
        const recorder = new MediaRecorder(micStreamRef.current, { mimeType });

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };

        recorder.onstop = async () => {
          const capturedPeak = slicePeakLevel;
          const avgRms = sliceSamples > 0 ? sliceRmsAccum / sliceSamples : 0;
          if (micRecorderActiveRef.current) {
            recordNextSlice();
          }
          if (chunks.length === 0 || (capturedPeak < 0.032 && avgRms < 0.012)) {
            return;
          }

          const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
          try {
            const buf = await blob.arrayBuffer();
            const audioBase64 = arrayBufferToBase64(buf);
            const resp = await fetch('/api/transcribe-chunk', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                audioBase64,
                mimeType: mimeType.split(';')[0],
                previousLyric: lastMicPhraseRef.current || undefined,
              }),
            });
            const data = (await resp.json()) as {
              error?: string;
              hasLyrics?: boolean;
              lyricLine?: string;
            };
            if (!resp.ok) {
              throw new Error(data.error || '即時辨識失敗');
            }
            const nextText = (data.lyricLine ?? '').trim();
            if (
              data.hasLyrics &&
              nextText &&
              normalizePhrase(nextText) !== normalizePhrase(lastMicPhraseRef.current)
            ) {
              lastMicPhraseRef.current = nextText;
              setCurrentLyricText(nextText);
              onLyricRef.current(nextText, 0.54);
            }
          } catch (err) {
            setErrorMsg(formatFriendlyError(err, 'mic'));
          }
        };

        recorder.start();
        window.setTimeout(() => {
          if (recorder.state === 'recording') {
            recorder.stop();
          }
        }, 2500);
      };

      recordNextSlice();
    } catch (err) {
      setIsMicListening(false);
      setDetectionStage('error');
      setErrorMsg(formatFriendlyError(err, 'mic'));
    }
  }, [startSyncLoop, stopAudioFile, stopTextQueue]);

  const writeManualLine = useCallback((text: string) => {
    const clean = text.trim();
    if (!clean) return;
    setErrorMsg(null);
    setCurrentLyricText(clean);
    onLyricRef.current(clean, 0.58, false);
  }, []);

  const startTextQueue = useCallback(
    (rawBlock: string, intervalSec = 3.2) => {
      stopTextQueue();
      const lines = rawBlock
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean);
      if (lines.length === 0) return;

      setErrorMsg(null);
      setIsScriptPlaying(true);
      let idx = 0;
      const emit = () => {
        const line = lines[idx];
        setCurrentLyricText(line);
        onLyricRef.current(line, 0.58, false);
        idx += 1;
        if (idx >= lines.length) {
          idx = 0;
        }
      };
      emit();
      if (lines.length > 1) {
        scriptTimerRef.current = window.setInterval(
          emit,
          Math.max(1800, intervalSec * 1000),
        );
      } else {
        setIsScriptPlaying(false);
      }
    },
    [stopTextQueue],
  );

  const getRecordingAudioStream = useCallback((): MediaStream | null => {
    if (sourceKind === 'mic' && micStreamRef.current) {
      return micStreamRef.current;
    }
    if (sourceKind === 'file' && recordDestRef.current) {
      return recordDestRef.current.stream;
    }
    return null;
  }, [sourceKind]);

  useEffect(() => {
    return () => {
      stopMic();
      stopTextQueue();
      stopAudioFile();
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, [stopAudioFile, stopMic, stopTextQueue]);

  return {
    sourceKind,
    detectionStage,
    statusDetail,
    songTitle,
    bpm,
    mood,
    isAnalyzing,
    isPlaying,
    isMicListening,
    isScriptPlaying,
    currentTime,
    duration,
    lyrics,
    activeLineIndex,
    currentLyricText,
    hasVocals,
    loopTrack,
    playbackRate,
    lyricOffsetSec,
    errorMsg,
    hasLoadedFile: Boolean(stableBlobRef.current),
    setLoopTrack,
    setPlaybackRate,
    setLyricOffsetSec,
    loadAudioFile,
    retryAnalysis,
    alignPlainLyricsWithAudio,
    importSyncedLyricsFile,
    togglePlayPause,
    seekTo,
    jumpToLyricLine,
    startMicListening,
    stopMic,
    writeManualLine,
    startTextQueue,
    stopTextQueue,
    getRecordingAudioStream,
  };
}

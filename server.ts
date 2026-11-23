import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import multer from 'multer';
import {
  GoogleGenAI,
  ThinkingLevel,
  Type,
  createPartFromUri,
  type Part,
} from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const isProduction = process.env.NODE_ENV === 'production';

// Abuse protection (all overridable through environment variables)
const MAX_AUDIO_MB = Number(process.env.MAX_AUDIO_MB || 200);
const RATE_ANALYZE_PER_MIN = Number(process.env.RATE_ANALYZE_PER_MIN || 10);
const RATE_TRANSCRIBE_PER_MIN = Number(process.env.RATE_TRANSCRIBE_PER_MIN || 60);
const RATE_STROKES_PER_MIN = Number(process.env.RATE_STROKES_PER_MIN || 240);
// Number of reverse proxies in front of the app (Cloud Run = 1). Needed so req.ip is the real client.
const TRUST_PROXY_HOPS = Number(process.env.TRUST_PROXY_HOPS ?? 1);

// Minimal in-memory sliding-window rate limiter (per client IP)
function createRateLimiter(maxPerMinute: number): express.RequestHandler {
  const windowMs = 60_000;
  const hits = new Map<string, number[]>();
  setInterval(() => {
    const cutoff = Date.now() - windowMs;
    for (const [key, times] of hits) {
      const recent = times.filter((t) => t > cutoff);
      if (recent.length > 0) hits.set(key, recent);
      else hits.delete(key);
    }
  }, windowMs).unref();

  return (req, res, next) => {
    const key = req.ip || 'unknown';
    const now = Date.now();
    const recent = (hits.get(key) ?? []).filter((t) => t > now - windowMs);
    if (recent.length >= maxPerMinute) {
      res.setHeader('Retry-After', '60');
      res.status(429).json({ error: '請求太頻繁，請稍後再試。' });
      return;
    }
    recent.push(now);
    hits.set(key, recent);
    next();
  };
}

// Disk storage for large audio tracks (WAV, FLAC, M4A, MP3, MP4, etc.), capped by MAX_AUDIO_MB
const uploadDir = path.join(os.tmpdir(), 'fog-mirror-audio');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const audioUpload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadDir),
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname || '').toLowerCase() || '.bin';
      cb(null, `track-${Date.now()}-${Math.random().toString(36).slice(2, 8)}${ext}`);
    },
  }),
  limits: { fileSize: MAX_AUDIO_MB * 1024 * 1024, files: 1 },
});

function resolveAudioMimeType(declaredMime: string | undefined, fileName: string): string {
  const ext = path.extname(fileName || '').toLowerCase();
  const extMap: Record<string, string> = {
    '.mp3': 'audio/mp3',
    '.wav': 'audio/wav',
    '.wave': 'audio/wav',
    '.m4a': 'audio/mp4',
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.aac': 'audio/aac',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.oga': 'audio/ogg',
    '.opus': 'audio/ogg',
    '.webm': 'audio/webm',
    '.aiff': 'audio/aiff',
    '.aif': 'audio/aiff',
    '.caf': 'audio/x-caf',
    '.3gp': 'audio/3gpp',
  };
  if (ext && extMap[ext]) return extMap[ext];
  if (declaredMime && declaredMime !== 'application/octet-stream') {
    return declaredMime.split(';')[0];
  }
  return 'audio/mpeg';
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('伺服器尚未設定 GEMINI_API_KEY，請於 Secrets 面板確認金鑰設定。');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function waitForFileActive(ai: GoogleGenAI, fileName: string) {
  let info = await ai.files.get({ name: fileName });
  let attempts = 0;
  while (info.state === 'PROCESSING' && attempts < 45) {
    await new Promise((r) => setTimeout(r, 1200));
    info = await ai.files.get({ name: fileName });
    attempts += 1;
  }
  if (info.state === 'FAILED') {
    throw new Error('音訊檔案雲端處理失敗，請確認檔案格式是否完整。');
  }
  return info;
}

const hanziDataDir = (() => {
  try {
    return path.dirname(path.resolve(__dirname, 'node_modules/hanzi-writer-data/package.json'));
  } catch {
    return path.resolve(__dirname, 'node_modules/hanzi-writer-data');
  }
})();
const hanziMedianCache = new Map<string, number[][][] | null>();

async function loadHanziMedians(ch: string): Promise<number[][][] | null> {
  if (hanziMedianCache.has(ch)) {
    return hanziMedianCache.get(ch) ?? null;
  }
  if (!/^[\u3400-\u9FFF\uF900-\uFAFF]$/.test(ch)) {
    hanziMedianCache.set(ch, null);
    return null;
  }
  const filePath = path.join(hanziDataDir, `${ch}.json`);
  try {
    const raw = await fs.promises.readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw) as { medians?: number[][][] };
    if (!Array.isArray(parsed.medians) || parsed.medians.length === 0) {
      hanziMedianCache.set(ch, null);
      return null;
    }
    const normalized = parsed.medians
      .filter((stroke) => Array.isArray(stroke) && stroke.length >= 2)
      .map((stroke) =>
        stroke.map(([mx, my]) => [
          Number((mx / 1024 - 0.5).toFixed(3)),
          Number(((900 - my) / 1024 - 0.5).toFixed(3)),
        ]),
      );
    if (hanziMedianCache.size > 4000) hanziMedianCache.clear();
    hanziMedianCache.set(ch, normalized);
    return normalized;
  } catch {
    hanziMedianCache.set(ch, null);
    return null;
  }
}

async function startServer() {
  const app = express();
  app.set('trust proxy', TRUST_PROXY_HOPS);

  // Per-route body limits instead of one global 50mb parser
  const jsonSmall = express.json({ limit: '1mb' });
  const jsonChunk = express.json({ limit: '4mb' });
  const jsonLargeAudio = express.json({ limit: '30mb' }); // base64 fallback path only

  // Endpoint 0: Instant local CJK/Kanji stroke-order medians lookup from hanzi-writer-data
  app.post('/api/stroke-medians', createRateLimiter(RATE_STROKES_PER_MIN), jsonSmall, async (req, res) => {
    const charsInput = Array.isArray(req.body?.chars) ? (req.body.chars as unknown[]) : [];
    const uniqueChars = Array.from(
      new Set(
        charsInput
          .filter((c): c is string => typeof c === 'string' && c.length > 0)
          .slice(0, 80),
      ),
    );
    const medians: Record<string, number[][][]> = {};
    await Promise.all(
      uniqueChars.map(async (ch) => {
        const m = await loadHanziMedians(ch);
        if (m) medians[ch] = m;
      }),
    );
    res.json({ medians });
  });

  // Endpoint 1: Analyze uploaded music of ANY size for precision timestamped lyrics
  app.post(
    '/api/analyze-lyrics',
    createRateLimiter(RATE_ANALYZE_PER_MIN),
    jsonLargeAudio,
    audioUpload.single('audio'),
    async (req, res) => {
    const tempFilePath = req.file?.path;
    let uploadedRemoteName: string | undefined;

    try {
      const ai = getGeminiClient();
      const plainLyrics =
        typeof req.body?.plainLyrics === 'string'
          ? req.body.plainLyrics.trim()
          : '';

      let audioPart: Part;
      let fileName = '未知音樂';
      let mimeType = 'audio/mpeg';

      if (req.file) {
        fileName = req.file.originalname || '已上傳音樂';
        mimeType = resolveAudioMimeType(req.file.mimetype, fileName);

        // Use inlineData for smaller tracks (<= 10MB) for minimal latency;
        // stream unlimited-size files (> 10MB) cleanly via Gemini Files API.
        if (req.file.size <= 10 * 1024 * 1024) {
          const buf = await fs.promises.readFile(req.file.path);
          audioPart = {
            inlineData: {
              mimeType,
              data: buf.toString('base64'),
            },
          };
        } else {
          const uploaded = await ai.files.upload({
            file: req.file.path,
            config: {
              mimeType,
              displayName: fileName,
            },
          });
          uploadedRemoteName = uploaded.name;
          const activeFile = uploaded.name
            ? await waitForFileActive(ai, uploaded.name)
            : uploaded;
          if (!activeFile.uri) {
            throw new Error('無法取得音訊檔案 URI。');
          }
          audioPart = createPartFromUri(
            activeFile.uri,
            activeFile.mimeType || mimeType,
          );
        }
      } else {
        // Fallback JSON support
        const { audioBase64, mimeType: bodyMime, fileName: bodyName } = req.body as {
          audioBase64?: string;
          mimeType?: string;
          fileName?: string;
        };
        if (!audioBase64 || !bodyMime) {
          res.status(400).json({ error: '缺少音訊檔案資料。' });
          return;
        }
        fileName = bodyName || '已上傳音樂';
        mimeType = resolveAudioMimeType(bodyMime, fileName);
        audioPart = {
          inlineData: {
            mimeType,
            data: audioBase64,
          },
        };
      }

      const alignmentGuidance = plainLyrics
        ? `\n使用者已提供參考歌詞文本如下，請優先將這份歌詞準確對齊到音樂實際演唱的每一句開始與結束秒數：\n---\n${plainLyrics}\n---\n`
        : '';

      const promptText = `你是一位頂尖音樂剪輯師與多語系聽寫對拍工程師。請全神貫注聆聽整段音訊（檔名參考：${fileName}），精準辨識演唱歌詞與每一句真正開口的拍點時間。${alignmentGuidance}
支援語言：韓文（한국어 / Hangul）、日文（日本語 / 漢字與假名）、繁體中文、英文及混語演唱。
嚴格要求：
1. 忠實保留原曲實際演唱語言與文字（韓文輸出韓文諺文、日文輸出日文漢字與假名、中文輸出繁體中文、英文輸出英文），嚴禁翻譯、簡化或轉成羅馬拼音。
2. 以小數點後兩位（例如 14.32）精準標記歌手「真正唱出第一個字」的 startTimeSec 與「該樂句收尾」的 endTimeSec；前奏、間奏或純配樂空檔不要提前插入歌詞。
3. 為了讓字體在霧氣鏡面上大字呈現且不擁擠，每一句務必精煉斷句：中/日/韓文每句控制在 3～12 字以內，英文每句控制在 2～6 個單字以內；長句請依換氣拍點自然拆分成多句。
4. 若為副歌或情緒張力最強段落，請標記 isChorus: true，並估算曲速 bpm 與音樂氛圍 mood（繁體中文簡述）。
5. 若整首皆為純音樂演奏無人聲，請將 hasVocals 設為 false 並回傳空的 lines 陣列，切勿捏造歌詞。`;

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts: [audioPart, { text: promptText }] },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              songTitle: {
                type: Type.STRING,
                description: '辨識到的歌曲或音訊名稱摘要',
              },
              hasVocals: {
                type: Type.BOOLEAN,
                description: '是否偵測到人聲演唱或口白歌詞',
              },
              language: {
                type: Type.STRING,
                description: '主要歌詞語言（例如 ko, ja, zh-Hant, en）',
              },
              bpm: {
                type: Type.NUMBER,
                description: '估算曲速 BPM',
              },
              mood: {
                type: Type.STRING,
                description: '簡短音樂氛圍描述（繁體中文）',
              },
              lines: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    startTimeSec: {
                      type: Type.NUMBER,
                      description: '該句歌詞開口的精確秒數',
                    },
                    endTimeSec: {
                      type: Type.NUMBER,
                      description: '該句歌詞結束的精確秒數',
                    },
                    text: {
                      type: Type.STRING,
                      description: '原始語言的精煉單句歌詞內容',
                    },
                    isChorus: {
                      type: Type.BOOLEAN,
                      description: '是否為副歌或高潮張力樂句',
                    },
                  },
                },
              },
            },
          },
        },
      });

      const raw = (response.text ?? '{}').trim();
      const parsed = JSON.parse(raw) as {
        songTitle?: string;
        hasVocals?: boolean;
        language?: string;
        bpm?: number;
        mood?: string;
        lines?: Array<{
          startTimeSec?: number;
          endTimeSec?: number;
          text?: string;
          isChorus?: boolean;
        }>;
      };

      const cleanLines = (parsed.lines ?? [])
        .filter((l) => typeof l.text === 'string' && l.text.trim().length > 0)
        .map((l) => ({
          startTimeSec: Math.max(0, Number(l.startTimeSec ?? 0)),
          endTimeSec: Math.max(
            Number(l.startTimeSec ?? 0) + 0.8,
            Number(l.endTimeSec ?? 0),
          ),
          text: String(l.text).trim(),
          isChorus: Boolean(l.isChorus),
        }))
        .sort((a, b) => a.startTimeSec - b.startTimeSec);

      res.json({
        songTitle: parsed.songTitle || fileName.replace(/\.[^.]+$/, '') || '已載入音樂',
        hasVocals: Boolean(parsed.hasVocals && cleanLines.length > 0),
        language: parsed.language || '未知',
        bpm: Number(parsed.bpm ?? 0) || undefined,
        mood: parsed.mood || undefined,
        lines: cleanLines,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : '分析音樂歌詞時發生未知錯誤。';
      console.error('[analyze-lyrics]', err);
      res.status(500).json({ error: message });
    } finally {
      if (tempFilePath) {
        fs.promises.unlink(tempFilePath).catch(() => {});
      }
      if (uploadedRemoteName) {
        try {
          const ai = getGeminiClient();
          await ai.files.delete({ name: uploadedRemoteName });
        } catch {
          // Ignore remote cleanup error
        }
      }
    }
    },
  );

  // Endpoint 2: Low-latency live microphone music/vocal snippet detection (Supports KR, JP, TC, EN)
  app.post('/api/transcribe-chunk', createRateLimiter(RATE_TRANSCRIBE_PER_MIN), jsonChunk, async (req, res) => {
    try {
      const { audioBase64, mimeType, previousLyric } = req.body as {
        audioBase64?: string;
        mimeType?: string;
        previousLyric?: string;
      };

      if (!audioBase64 || !mimeType) {
        res.status(400).json({ error: '缺少麥克風音訊片段。' });
        return;
      }

      const ai = getGeminiClient();

      const audioPart = {
        inlineData: {
          mimeType: mimeType.split(';')[0],
          data: audioBase64,
        },
      };

      const contextHint = previousLyric
        ? `前一句剛出現的歌詞為「${previousLyric}」，請優先聽寫緊接著唱出的「最新完整短句」，避免重複輸出上一句。`
        : '';

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            audioPart,
            {
              text: `請仔細聆聽這段約 2.5 秒的即時收音片段（可能播放韓文、日文、中文或英文音樂／清唱）。${contextHint}
規則：
1. 若清楚聽到歌詞或人聲語句，務必保留原曲語言文字（韓文輸出韓文諺文、日文輸出日文漢字與假名、中文輸出繁體中文、英文輸出英文，切勿翻譯或轉拼音）。
2. 回傳精煉清楚的單句短詞（中/日/韓文約 2～12 字、英文約 2～6 字），適合在霧鏡上大字手寫。
3. 若只有純音樂伴奏、環境雜音、模糊氣音或安靜無聲，請將 hasLyrics 設為 false、lyricLine 留空字串，嚴禁臆測或編造。`,
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              hasLyrics: {
                type: Type.BOOLEAN,
                description: '是否清楚聽到歌詞或人聲語句',
              },
              lyricLine: {
                type: Type.STRING,
                description: '聽到的原始語言最新一句精煉歌詞',
              },
            },
          },
        },
      });

      const raw = (response.text ?? '{}').trim();
      const parsed = JSON.parse(raw) as {
        hasLyrics?: boolean;
        lyricLine?: string;
      };

      res.json({
        hasLyrics: Boolean(
          parsed.hasLyrics && parsed.lyricLine && parsed.lyricLine.trim(),
        ),
        lyricLine: (parsed.lyricLine ?? '').trim(),
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : '即時歌詞偵測失敗。';
      console.error('[transcribe-chunk]', err);
      res.status(500).json({ error: message });
    }
  });

  // Turn upload / body-size / parse errors into clean JSON instead of HTML stack pages
  app.use(
    (
      err: unknown,
      _req: express.Request,
      res: express.Response,
      next: express.NextFunction,
    ) => {
      if (err instanceof multer.MulterError) {
        const tooBig = err.code === 'LIMIT_FILE_SIZE';
        res.status(tooBig ? 413 : 400).json({
          error: tooBig ? `音訊檔案超過 ${MAX_AUDIO_MB}MB 上限。` : '音訊上傳失敗。',
        });
        return;
      }
      const status =
        typeof err === 'object' && err !== null && 'status' in err
          ? Number((err as { status?: unknown }).status)
          : 0;
      if (status >= 400 && status < 500) {
        res.status(status).json({ error: status === 413 ? '上傳內容過大。' : '請求格式錯誤。' });
        return;
      }
      next(err);
    },
  );

  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      const indexFile = path.join(distDir, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Build artifacts not found.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});

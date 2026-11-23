export type MirrorSource = 'studio' | 'camera' | 'media';

export type StudioTheme = 'daylight' | 'velvet_slate';

export type FogPhase = 'fogging' | 'saturated' | 'clearing' | 'clear';

export type FogCoveragePreset = 'edge' | 'halo' | 'full';

export interface LyricLine {
  startTimeSec: number;
  endTimeSec: number;
  text: string;
  isChorus?: boolean;
}

export interface MirrorSettings {
  breathCycle: boolean; // false = continuous natural condensation (default); true = optional periodic breath wave
  recondenseSpeed: number; // 0.6 to 3.2 (how fast wiped trails & glass re-condense)
  maxDensity: number; // 0.25 to 1.0
  fogCoverage: number; // 0.22 to 1.0
  tempDiff: number; // 0.2 to 1.0 (glass-to-steam temperature differential ΔT)
  mirrorSource: MirrorSource;
  studioTheme: StudioTheme;
  lyricsActive: boolean;
  handwritingScale: number; // 0.7 to 1.4
  strokeWeight: number; // 0.15 (ultra-fine fingertip) to 1.6 (broad finger wipe)
  beatReactive: boolean;
}

export interface FogStatus {
  fogLevel: number; // 0 to 1
  phase: FogPhase;
  wipedRatio: number; // 0 to 1
}

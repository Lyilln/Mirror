import {
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  forwardRef,
} from 'react';
import * as THREE from 'three';
import {
  BLUR_FRAGMENT_SHADER,
  FLUID_NORMAL_FRAGMENT_SHADER,
  STROKE_DEPTH_FRAGMENT_SHADER,
  createFogSimulation,
} from '../mirror/fogSimulation';
import { createReflectionWorld } from '../mirror/reflectionWorld';
import type { FogStatus, MirrorSettings } from '../mirror/types';

export interface MirrorViewportHandle {
  triggerSteamBurst: () => void;
  clearAllFog: () => void;
  writeLyricOnGlass: (
    text: string,
    durationSec?: number,
    isChorus?: boolean,
  ) => void;
  setAudioEnergy: (energy: number) => void;
  triggerBeatPulse: (intensity?: number) => void;
  getCanvasElement: () => HTMLCanvasElement | null;
}

interface MirrorViewportProps {
  settings: MirrorSettings;
  customMediaUrl: string | null;
  customMediaType: 'image' | 'video' | null;
  onFogStatusChange: (status: FogStatus) => void;
  onCameraError: (msg: string | null) => void;
  onCameraFallback: () => void;
  onDoubleTapGlass?: () => void;
}

const QUAD_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Performance switches.
// Open the app with ?legacy=1 to restore the original behaviour, so the new and
// old settings can be compared side by side on a real device.
// ---------------------------------------------------------------------------
const LEGACY =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('legacy') === '1';

const PERF = {
  // Final output is one full-screen quad: MSAA and depth buffers buy nothing.
  antialias: LEGACY,
  depthBuffer: LEGACY,
  // Cap pixel ratio on touch devices (3x phones render 4x the pixels of 1.5x).
  mobileDprCap: LEGACY ? 2 : 1.5,
  desktopDprCap: 2,
  // Fluid-normal / stroke-depth passes read <=1440px textures; they do not need 0.85x of screen res.
  fluidScale: LEGACY ? 0.85 : 0.6,
  // 0 = uncapped. Stops 120Hz phones from doing twice the GPU work.
  maxFps: LEGACY ? 0 : 60,
  // Studio reflection + blur chain only re-runs when parallax / theme / audio changes.
  cacheReflection: !LEGACY,
  // Fluid passes run every N frames (and not at all when nothing changed).
  fluidEveryNFrames: LEGACY ? 1 : 2,
  resizeDebounceMs: LEGACY ? 0 : 120,
  cameraIdealW: LEGACY ? 1920 : 1280,
  cameraIdealH: LEGACY ? 1080 : 720,
};

// Frames to keep redrawing after the last change so the damped uniforms can settle.
const REFLECT_SETTLE_FRAMES = 80;

function detectSoftwareWebGL(): boolean {
  try {
    const probeCanvas = document.createElement('canvas');
    const gl = probeCanvas.getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

export const MirrorViewport = forwardRef<
  MirrorViewportHandle,
  MirrorViewportProps
>(function MirrorViewport(
  {
    settings,
    customMediaUrl,
    customMediaType,
    onFogStatusChange,
    onCameraError,
    onCameraFallback,
    onDoubleTapGlass,
  },
  ref,
) {
  const mountRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<MirrorSettings>(settings);
  const statusCallbackRef = useRef(onFogStatusChange);
  const cameraErrorRef = useRef(onCameraError);
  const cameraFallbackRef = useRef(onCameraFallback);
  const doubleTapRef = useRef(onDoubleTapGlass);

  const fogSystemRef = useRef<ReturnType<typeof createFogSimulation> | null>(
    null,
  );
  const reflectionWorldRef = useRef<ReturnType<
    typeof createReflectionWorld
  > | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const customVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasElRef = useRef<HTMLCanvasElement | null>(null);
  const audioEnergyRef = useRef<number>(0);

  const [webglLost, setWebglLost] = useState(false);

  settingsRef.current = settings;
  statusCallbackRef.current = onFogStatusChange;
  cameraErrorRef.current = onCameraError;
  cameraFallbackRef.current = onCameraFallback;
  doubleTapRef.current = onDoubleTapGlass;

  // Manage live mirrored camera feed when mirrorSource === 'camera'
  useEffect(() => {
    let cancelled = false;

    const stopCamera = () => {
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((t) => t.stop());
        cameraStreamRef.current = null;
      }
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = null;
        cameraVideoRef.current = null;
      }
      reflectionWorldRef.current?.setCameraStream(null);
    };

    if (settings.mirrorSource !== 'camera') {
      stopCamera();
      cameraErrorRef.current(null);
      return;
    }

    const startCamera = async () => {
      try {
        cameraErrorRef.current(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: PERF.cameraIdealW },
            height: { ideal: PERF.cameraIdealH },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        cameraStreamRef.current = stream;
        const video = document.createElement('video');
        video.playsInline = true;
        video.muted = true;
        video.autoplay = true;
        video.srcObject = stream;
        await video.play();
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        cameraVideoRef.current = video;
        reflectionWorldRef.current?.setCameraStream(video);
      } catch (err) {
        if (!cancelled) {
          cameraErrorRef.current(
            err instanceof Error
              ? '無法取得前置鏡頭權限，已切回光影反射空間。'
              : '無法開啟鏡頭，已切回光影反射空間。',
          );
          cameraFallbackRef.current();
        }
      }
    };

    void startCamera();

    return () => {
      cancelled = true;
      stopCamera();
    };
  }, [settings.mirrorSource]);

  // Manage custom uploaded image/video backdrop for video editing
  useEffect(() => {
    let cancelled = false;

    const cleanupMedia = () => {
      if (customVideoRef.current) {
        customVideoRef.current.pause();
        customVideoRef.current.src = '';
        customVideoRef.current = null;
      }
      reflectionWorldRef.current?.setCustomMedia(null);
    };

    if (!customMediaUrl || !customMediaType) {
      cleanupMedia();
      return;
    }

    if (customMediaType === 'image') {
      const img = new Image();
      img.onload = () => {
        if (!cancelled) {
          reflectionWorldRef.current?.setCustomMedia(img);
        }
      };
      img.src = customMediaUrl;
    } else {
      const vid = document.createElement('video');
      vid.src = customMediaUrl;
      vid.loop = true;
      vid.muted = true;
      vid.playsInline = true;
      vid.autoplay = true;
      vid.onloadeddata = () => {
        if (!cancelled) {
          void vid.play();
          customVideoRef.current = vid;
          reflectionWorldRef.current?.setCustomMedia(vid);
        }
      };
    }

    return () => {
      cancelled = true;
      cleanupMedia();
    };
  }, [customMediaUrl, customMediaType]);

  useImperativeHandle(ref, () => ({
    triggerSteamBurst: () => {
      fogSystemRef.current?.triggerSteamBurst();
    },
    clearAllFog: () => {
      fogSystemRef.current?.clearAllFog();
    },
    writeLyricOnGlass: (text: string, durationSec?: number, isChorus?: boolean) => {
      fogSystemRef.current?.writeLyricOnGlass(text, durationSec, isChorus);
    },
    setAudioEnergy: (energy: number) => {
      audioEnergyRef.current = energy;
    },
    triggerBeatPulse: (intensity?: number) => {
      fogSystemRef.current?.triggerBeatPulse(intensity);
    },
    getCanvasElement: () => canvasElRef.current,
  }));

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isSoftwareGl = detectSoftwareWebGL();
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;
    const isCoarsePointer =
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(pointer: coarse)').matches;
    const dprCap = isCoarsePointer ? PERF.mobileDprCap : PERF.desktopDprCap;
    const dpr = isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, dprCap);

    const renderer = new THREE.WebGLRenderer({
      antialias: PERF.antialias && !isSoftwareGl,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.14;
    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.touchAction = 'none';

    canvasElRef.current = renderer.domElement;
    container.appendChild(renderer.domElement);

    // Cache / throttle state (declared early so the context-restore handler can reset it)
    let forceReflection = true;
    let reflectSettleFrames = 0;
    let lastReflectKey = '';
    let lastReflectParX = Infinity;
    let lastReflectParY = Infinity;
    let lastReflectEnergy = Infinity;
    let fluidForce = true;
    let fluidFrame = 0;
    let lastFluidParX = Infinity;
    let lastTickTs = -1;
    const minFrameMs = PERF.maxFps > 0 ? 1000 / PERF.maxFps - 1.5 : 0;

    const handleContextLost = (e: Event) => {
      e.preventDefault();
      setWebglLost(true);
    };
    const handleContextRestored = () => {
      forceReflection = true;
      fluidForce = true;
      setWebglLost(false);
    };
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost);
    renderer.domElement.addEventListener(
      'webglcontextrestored',
      handleContextRestored,
    );

    const makeRT = (w: number, h: number) =>
      new THREE.WebGLRenderTarget(w, h, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        depthBuffer: PERF.depthBuffer,
      });

    const rtW = Math.max(256, Math.round(width * dpr));
    const rtH = Math.max(256, Math.round(height * dpr));
    const blurW = Math.max(192, Math.round(rtW * 0.5));
    const blurH = Math.max(192, Math.round(rtH * 0.5));

    const sharpRT = makeRT(rtW, rtH);
    const nearBlurRT = makeRT(blurW, blurH);
    const blurTempRT = makeRT(blurW, blurH);
    const blurFinalRT = makeRT(blurW, blurH);
    const fluidW = Math.max(256, Math.round(rtW * PERF.fluidScale));
    const fluidH = Math.max(256, Math.round(rtH * PERF.fluidScale));
    const fluidNormalRT = makeRT(fluidW, fluidH);
    const strokeDepthRT = makeRT(fluidW, fluidH);

    // Separable Gaussian Blur Pass Scene
    const blurScene = new THREE.Scene();
    const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quadGeo = new THREE.PlaneGeometry(2, 2);
    const blurMat = new THREE.ShaderMaterial({
      uniforms: {
        uTex: { value: null },
        uTexelOffset: { value: new THREE.Vector2(1 / blurW, 0) },
      },
      vertexShader: QUAD_VERTEX,
      fragmentShader: BLUR_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
    });
    const blurQuad = new THREE.Mesh(quadGeo, blurMat);
    blurScene.add(blurQuad);

    const reflectionWorld = createReflectionWorld(isSoftwareGl);
    reflectionWorldRef.current = reflectionWorld;
    if (cameraVideoRef.current) {
      reflectionWorld.setCameraStream(cameraVideoRef.current);
    }

    const fogSim = createFogSimulation();
    fogSystemRef.current = fogSim;
    fogSim.setTextures(
      sharpRT.texture,
      nearBlurRT.texture,
      blurFinalRT.texture,
      fluidNormalRT.texture,
      strokeDepthRT.texture,
    );
    fogSim.setMirrorAspect(width / Math.max(1, height));

    // Dedicated Liquid Mechanics & High-Precision Normal Map Pass
    const fluidScene = new THREE.Scene();
    const fluidMat = new THREE.ShaderMaterial({
      uniforms: {
        uWipeMap: { value: fogSim.fluidInputs.wipeMap },
        uDropletMap: { value: fogSim.fluidInputs.dropletMap },
        uWetFilmMap: { value: fogSim.fluidInputs.wetFilmMap },
        uWipeTexSize: { value: fogSim.fluidInputs.wipeTexSize },
        uDropTexSize: { value: fogSim.fluidInputs.dropTexSize },
        uWetTexSize: { value: fogSim.fluidInputs.wetTexSize },
        uGravityVec: { value: new THREE.Vector2(0, -1) },
        uTime: { value: 0 },
      },
      vertexShader: QUAD_VERTEX,
      fragmentShader: FLUID_NORMAL_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
    });
    const fluidQuad = new THREE.Mesh(quadGeo, fluidMat);
    fluidScene.add(fluidQuad);

    // Stroke Indentation Depth Buffer & Curvature Tension Pass
    const strokeDepthScene = new THREE.Scene();
    const strokeDepthMat = new THREE.ShaderMaterial({
      uniforms: {
        uWipeMap: { value: fogSim.fluidInputs.wipeMap },
        uDropletMap: { value: fogSim.fluidInputs.dropletMap },
        uWetFilmMap: { value: fogSim.fluidInputs.wetFilmMap },
        uWipeTexSize: { value: fogSim.fluidInputs.wipeTexSize },
        uDropTexSize: { value: fogSim.fluidInputs.dropTexSize },
        uWetTexSize: { value: fogSim.fluidInputs.wetTexSize },
      },
      vertexShader: QUAD_VERTEX,
      fragmentShader: STROKE_DEPTH_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
    });
    const strokeDepthQuad = new THREE.Mesh(quadGeo, strokeDepthMat);
    strokeDepthScene.add(strokeDepthQuad);

    const screenScene = new THREE.Scene();
    const screenQuad = new THREE.Mesh(quadGeo, fogSim.screenMaterial);
    screenScene.add(screenQuad);
    screenScene.add(fogSim.steamParticles);

    let isWipingPointer = false;
    let pointerDownPos: { x: number; y: number; t: number } | null = null;
    let lastQuickTap: { x: number; y: number; t: number } | null = null;
    let targetWipeUv: { u: number; v: number } | null = null;
    let springWipeUv: { u: number; v: number } | null = null;
    let springVel = { u: 0, v: 0 };
    let lastFogDensity = 0.85;
    let parallaxX = 0;
    let parallaxY = 0;

    const clientToUv = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const u = THREE.MathUtils.clamp(
        (e.clientX - rect.left) / Math.max(1, rect.width),
        0,
        1,
      );
      const v = THREE.MathUtils.clamp(
        1 - (e.clientY - rect.top) / Math.max(1, rect.height),
        0,
        1,
      );
      return { u, v };
    };

    const onPointerDown = (e: PointerEvent) => {
      isWipingPointer = true;
      const now = performance.now();
      pointerDownPos = { x: e.clientX, y: e.clientY, t: now };
      const uv = clientToUv(e);
      targetWipeUv = uv;
      springWipeUv = { ...uv };
      springVel = { u: 0, v: 0 };
      parallaxX = (uv.u - 0.5) * 2;
      parallaxY = (uv.v - 0.5) * 2;
      fogSim.wipeAtUv(uv.u, uv.v);
      renderer.domElement.setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      const uv = clientToUv(e);
      parallaxX = (uv.u - 0.5) * 2;
      parallaxY = (uv.v - 0.5) * 2;

      if (isWipingPointer) {
        targetWipeUv = uv;
      }
    };

    const endWipe = (e: PointerEvent) => {
      if (isWipingPointer) {
        isWipingPointer = false;
        if (springWipeUv && targetWipeUv) {
          fogSim.wipeStroke(
            springWipeUv.u,
            springWipeUv.v,
            targetWipeUv.u,
            targetWipeUv.v,
            0.032,
          );
        }
        targetWipeUv = null;
        springWipeUv = null;
        springVel = { u: 0, v: 0 };
        if (renderer.domElement.hasPointerCapture(e.pointerId)) {
          renderer.domElement.releasePointerCapture(e.pointerId);
        }

        if (pointerDownPos && e.type === 'pointerup') {
          const now = performance.now();
          const moved = Math.hypot(
            e.clientX - pointerDownPos.x,
            e.clientY - pointerDownPos.y,
          );
          if (moved < 16 && now - pointerDownPos.t < 260) {
            if (
              lastQuickTap &&
              now - lastQuickTap.t < 360 &&
              Math.hypot(e.clientX - lastQuickTap.x, e.clientY - lastQuickTap.y) < 36
            ) {
              lastQuickTap = null;
              doubleTapRef.current?.();
            } else {
              lastQuickTap = { x: e.clientX, y: e.clientY, t: now };
            }
          }
        }
        pointerDownPos = null;
      }
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerup', endWipe);
    renderer.domElement.addEventListener('pointercancel', endWipe);

    let currentW = width;
    let currentH = height;

    let resizeTimer = 0;
    const applyResize = () => {
      if (!container) return;
      const w = container.clientWidth || window.innerWidth;
      const h = container.clientHeight || window.innerHeight;
      currentW = w;
      currentH = h;
      renderer.setSize(w, h);

      const nextRtW = Math.max(256, Math.round(w * dpr));
      const nextRtH = Math.max(256, Math.round(h * dpr));
      const nextBlurW = Math.max(192, Math.round(nextRtW * 0.5));
      const nextBlurH = Math.max(192, Math.round(nextRtH * 0.5));

      sharpRT.setSize(nextRtW, nextRtH);
      nearBlurRT.setSize(nextBlurW, nextBlurH);
      blurTempRT.setSize(nextBlurW, nextBlurH);
      blurFinalRT.setSize(nextBlurW, nextBlurH);
      const nextFluidW = Math.max(256, Math.round(nextRtW * PERF.fluidScale));
      const nextFluidH = Math.max(256, Math.round(nextRtH * PERF.fluidScale));
      fluidNormalRT.setSize(nextFluidW, nextFluidH);
      strokeDepthRT.setSize(nextFluidW, nextFluidH);
      fogSim.setMirrorAspect(w / Math.max(1, h));
      // Resized render targets lose their contents: redraw everything once.
      forceReflection = true;
      fluidForce = true;
    };
    const handleResize = () => {
      if (PERF.resizeDebounceMs > 0) {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(applyResize, PERF.resizeDebounceMs);
      } else {
        applyResize();
      }
    };
    window.addEventListener('resize', handleResize);

    let frameId = 0;
    let disposed = false;
    let statusEmitTimer = 0;
    const clock = new THREE.Clock();

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) {
          if (sync) gl.deleteSync(sync);
          return;
        }
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    const runBlurPass = (
      sourceTex: THREE.Texture,
      destRT: THREE.WebGLRenderTarget,
      dx: number,
      dy: number,
    ) => {
      blurMat.uniforms.uTex.value = sourceTex;
      blurMat.uniforms.uTexelOffset.value.set(dx, dy);
      renderer.setRenderTarget(destRT);
      renderer.render(blurScene, orthoCam);
    };

    const tick = (now?: number) => {
      if (disposed) return;

      // Frame-rate cap (rAF path only)
      if (minFrameMs > 0 && now !== undefined) {
        if (lastTickTs >= 0 && now - lastTickTs < minFrameMs) {
          frameId = requestAnimationFrame(tick);
          return;
        }
        lastTickTs = now;
      }

      const dt = clock.getDelta();
      const clampedDt = Math.min(dt, 0.05);
      const elapsed = clock.getElapsedTime();
      const curSettings = settingsRef.current;
      const energy = curSettings.beatReactive ? audioEnergyRef.current : 0;
      const aspect = currentW / Math.max(1, currentH);

      // Spring-Damper Finger Kinematics Solver: viscous friction scales with surface fog density
      if (isWipingPointer && targetWipeUv && springWipeUv) {
        const fogFriction = 1 + lastFogDensity * 0.95;
        const stiffness = 360 / fogFriction;
        const damping = 28 * fogFriction;
        const ax = stiffness * (targetWipeUv.u - springWipeUv.u) - damping * springVel.u;
        const ay = stiffness * (targetWipeUv.v - springWipeUv.v) - damping * springVel.v;
        springVel.u += ax * clampedDt;
        springVel.v += ay * clampedDt;
        const nextU = THREE.MathUtils.clamp(springWipeUv.u + springVel.u * clampedDt, 0, 1);
        const nextV = THREE.MathUtils.clamp(springWipeUv.v + springVel.v * clampedDt, 0, 1);
        if (Math.hypot((nextU - springWipeUv.u) * aspect, nextV - springWipeUv.v) > 0.0003) {
          fogSim.wipeStroke(springWipeUv.u, springWipeUv.v, nextU, nextV, clampedDt);
        }
        springWipeUv.u = nextU;
        springWipeUv.v = nextV;
      }

      // Studio reflection + 6 blur passes only depend on parallax / theme / audio.
      // Camera and uploaded media change every frame, so they always redraw.
      let reflectionDirty = true;
      if (PERF.cacheReflection && curSettings.mirrorSource === 'studio') {
        const key = `${curSettings.studioTheme}|${aspect.toFixed(3)}`;
        const moved =
          Math.abs(parallaxX - lastReflectParX) > 0.004 ||
          Math.abs(parallaxY - lastReflectParY) > 0.004 ||
          Math.abs(energy - lastReflectEnergy) > 0.03;
        if (forceReflection || key !== lastReflectKey || moved) {
          forceReflection = false;
          lastReflectKey = key;
          lastReflectParX = parallaxX;
          lastReflectParY = parallaxY;
          lastReflectEnergy = energy;
          reflectSettleFrames = REFLECT_SETTLE_FRAMES;
        }
        reflectionDirty = reflectSettleFrames > 0;
        if (reflectSettleFrames > 0) reflectSettleFrames--;
      } else {
        // Leaving studio mode: make sure coming back redraws immediately
        lastReflectKey = '';
        forceReflection = true;
      }

      if (reflectionDirty) {
        reflectionWorld.renderSharpReflection(
          renderer,
          sharpRT,
          aspect,
          elapsed,
          energy,
          parallaxX,
          parallaxY,
          curSettings.mirrorSource,
          curSettings.studioTheme,
          clampedDt,
        );

        // Stage 1: Shallow Near-Glass DOF Blur -> nearBlurRT
        const bw = Math.max(1, blurTempRT.width);
        const bh = Math.max(1, blurTempRT.height);
        runBlurPass(sharpRT.texture, blurTempRT, 1.2 / bw, 0);
        runBlurPass(blurTempRT.texture, nearBlurRT, 0, 1.2 / bh);
        // Stage 2 & 3: Creamy Far-Field Volumetric Bokeh Blur -> blurFinalRT
        runBlurPass(nearBlurRT.texture, blurTempRT, 2.6 / bw, 0);
        runBlurPass(blurTempRT.texture, blurFinalRT, 0, 2.6 / bh);
        runBlurPass(blurFinalRT.texture, blurTempRT, 4.4 / bw, 0);
        runBlurPass(blurTempRT.texture, blurFinalRT, 0, 4.4 / bh);
        renderer.setRenderTarget(null);
      }

      const fogStatus = fogSim.update(
        dt,
        elapsed,
        curSettings.breathCycle,
        curSettings.recondenseSpeed,
        curSettings.maxDensity,
        curSettings.fogCoverage,
        curSettings.tempDiff,
        energy,
        curSettings.lyricsActive,
        curSettings.handwritingScale,
        curSettings.strokeWeight,
        parallaxX,
        parallaxY,
      );
      lastFogDensity = fogStatus.fogLevel;

      // Fluid passes: skipped while nothing wet/moving is on the glass, and run at
      // 1/N frame rate while droplets move (their motion is slow).
      if (Math.abs(parallaxX - lastFluidParX) > 0.01) {
        lastFluidParX = parallaxX;
        fluidForce = true;
      }
      let runFluid = true;
      if (PERF.fluidEveryNFrames > 1) {
        runFluid = false;
        if (fluidForce || fogSim.isFluidActive()) {
          fluidFrame++;
          runFluid = fluidForce || fluidFrame % PERF.fluidEveryNFrames === 0;
        }
      }
      fluidForce = false;

      if (runFluid) {
        // Dedicated Liquid Mechanics Pass: compute tangent-space liquid normals, droplet domes & meniscus thickness
        fluidMat.uniforms.uTime.value = elapsed;
        fluidMat.uniforms.uGravityVec.value.set(parallaxX * 0.22, -1.0);
        renderer.setRenderTarget(fluidNormalRT);
        renderer.render(fluidScene, orthoCam);

        // Stroke Indentation Depth Buffer & Curvature Tension Pass
        renderer.setRenderTarget(strokeDepthRT);
        renderer.render(strokeDepthScene, orthoCam);
        renderer.setRenderTarget(null);
      }

      statusEmitTimer += dt;
      if (statusEmitTimer > 0.1) {
        statusEmitTimer = 0;
        statusCallbackRef.current(fogStatus);
      }

      renderer.render(screenScene, orthoCam);

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    };

    if (isSoftwareGl) {
      tick();
    } else {
      frameId = requestAnimationFrame(tick);
    }

    return () => {
      disposed = true;
      canvasElRef.current = null;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      window.clearTimeout(resizeTimer);
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('pointerup', endWipe);
      renderer.domElement.removeEventListener('pointercancel', endWipe);
      renderer.domElement.removeEventListener(
        'webglcontextlost',
        handleContextLost,
      );
      renderer.domElement.removeEventListener(
        'webglcontextrestored',
        handleContextRestored,
      );
      fogSim.dispose();
      reflectionWorld.dispose();
      quadGeo.dispose();
      blurMat.dispose();
      fluidMat.dispose();
      strokeDepthMat.dispose();
      sharpRT.dispose();
      nearBlurRT.dispose();
      blurTempRT.dispose();
      blurFinalRT.dispose();
      fluidNormalRT.dispose();
      strokeDepthRT.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
      {webglLost && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0c0e13] text-[#e8e2d5] p-6">
          <div className="max-w-md text-center space-y-2 border border-white/10 bg-[#14171f] px-6 py-5 rounded-xl">
            <p className="font-display text-2xl">鏡面渲染暫時中斷</p>
            <p className="text-sm text-white/70">
              WebGL 圖形內容已重置，請重新整理頁面以恢復全螢幕霧氣鏡面效果。
            </p>
          </div>
        </div>
      )}
    </div>
  );
});

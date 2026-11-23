import * as THREE from 'three';
import type { StudioTheme } from './types';

export interface ReflectionWorld {
  renderSharpReflection: (
    renderer: THREE.WebGLRenderer,
    target: THREE.WebGLRenderTarget,
    aspect: number,
    elapsed: number,
    audioEnergy: number,
    parallaxX: number,
    parallaxY: number,
    mode: 'studio' | 'camera' | 'media',
    studioTheme: StudioTheme,
    dt?: number,
  ) => void;
  setCameraStream: (video: HTMLVideoElement | null) => void;
  setCustomMedia: (media: HTMLImageElement | HTMLVideoElement | null) => void;
  dispose: () => void;
}

const QUAD_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// Highest-Grade 3D Perspective Architectural Mirror World (Zero Mirror Frame, Zero Protagonist, Zero Night Sky)
// True 3D ray-casted room geometry with overhead skylight cove & deeper honed travertine floor/wall depth,
// providing physically authentic sky-above / rich-stone-below contrast for 3D Snell-refracted water beads.
const ARCHITECTURAL_REFLECTION_FRAGMENT = /* glsl */ `
  precision highp float;

  varying vec2 vUv;

  uniform float uAspect;
  uniform float uTime;
  uniform float uParallaxX;
  uniform float uParallaxY;
  uniform float uAudioEnergy;
  uniform float uTheme; // 0.0 = bright sunlit travertine daylight, 1.0 = deep slate daylight

  float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
      mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
      u.y
    );
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
      v += a * noise(p);
      p = p * 2.04 + vec2(3.1, 7.4);
      a *= 0.5;
    }
    return v;
  }

  float roundedBoxSdf(vec2 p, vec2 halfSize, float radius) {
    vec2 q = abs(p) - halfSize + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
  }

  void main() {
    // 3D Virtual Camera placed in front of the mirror looking into the reflected room
    vec2 ndc = (vUv - 0.5) * 2.0;
    vec3 ro = vec3(uParallaxX * 0.24, 0.04 + uParallaxY * 0.16, 1.35);
    vec3 rd = normalize(vec3(ndc.x * uAspect * 0.54 - uParallaxX * 0.08, ndc.y * 0.54 - uParallaxY * 0.06, -1.0));

    // 3D Room bounds in reflected space (behind glass z < 0):
    // Back wall z = -2.25, Ceiling y = +1.05, Floor/Vanity counter y = -0.82, Left window wall x = -1.65
    float tBack = (-2.25 - ro.z) / min(-0.05, rd.z);
    float tCeil = rd.y > 0.02 ? (1.05 - ro.y) / rd.y : 100.0;
    float tFloor = rd.y < -0.02 ? (-0.82 - ro.y) / rd.y : 100.0;
    float tLeft = rd.x < -0.02 ? (-1.55 - ro.x) / rd.x : 100.0;
    float tRight = rd.x > 0.02 ? (1.75 - ro.x) / rd.x : 100.0;

    float tHit = min(tBack, min(min(tCeil, tFloor), min(tLeft, tRight)));
    vec3 hitPos = ro + rd * tHit;

    // Select surface normal & 2D material coordinates
    vec2 matUv = hitPos.xy;
    vec3 surfN = vec3(0.0, 0.0, 1.0);
    float isCeiling = 0.0;
    float isFloor = 0.0;
    float isWindowWall = 0.0;

    if (tHit == tCeil) {
      isCeiling = 1.0;
      matUv = hitPos.xz;
      surfN = vec3(0.0, -1.0, 0.0);
    } else if (tHit == tFloor) {
      isFloor = 1.0;
      matUv = hitPos.xz;
      surfN = vec3(0.0, 1.0, 0.0);
    } else if (tHit == tLeft) {
      isWindowWall = 1.0;
      matUv = hitPos.zy;
      surfN = vec3(1.0, 0.0, 0.0);
    } else if (tHit == tRight) {
      matUv = hitPos.zy;
      surfN = vec3(-1.0, 0.0, 0.0);
    }

    // Subtle 3D architectural travertine & brushed plaster detail
    float vein = fbm(matUv * vec2(2.6, 4.2));
    float grain = fbm(matUv * 14.0);
    float stoneVar = 0.88 + vein * 0.16 + grain * 0.05;

    // Color Palette: Bright daylight skylight/window ABOVE vs rich honed architectural travertine/basalt BELOW
    // Deep enough behind glass (~0.24..0.44) so wiped mirror glass cuts sharply through pale white fog,
    // while bright overhead skylight (~0.88) powers authentic bright lower-belly caustics inside 3D droplets.
    vec3 skyLightDay = vec3(0.90, 0.93, 0.97);
    vec3 wallDay = mix(vec3(0.24, 0.27, 0.31), vec3(0.42, 0.41, 0.39), clamp(hitPos.y * 0.55 + 0.45, 0.0, 1.0));
    vec3 floorDay = vec3(0.14, 0.15, 0.16);

    vec3 skyLightSlate = vec3(0.72, 0.79, 0.88);
    vec3 wallSlate = mix(vec3(0.11, 0.14, 0.18), vec3(0.24, 0.28, 0.34), clamp(hitPos.y * 0.55 + 0.45, 0.0, 1.0));
    vec3 floorSlate = vec3(0.06, 0.08, 0.10);

    vec3 skyCol = mix(skyLightDay, skyLightSlate, uTheme);
    vec3 wallCol = mix(wallDay, wallSlate, uTheme) * stoneVar;
    vec3 floorCol = mix(floorDay, floorSlate, uTheme) * (0.90 + vein * 0.14);

    vec3 col = wallCol;
    if (isCeiling > 0.5) {
      // Recessed daylight skylight cove panel on ceiling
      float skyPanel = 1.0 - smoothstep(-0.05, 0.18, roundedBoxSdf(hitPos.xz - vec2(-0.15, -1.25), vec2(0.85, 0.72), 0.12));
      col = mix(wallCol * 1.15, skyCol * (1.0 + uAudioEnergy * 0.14), skyPanel * 0.94);
    } else if (isFloor > 0.5) {
      // Honed stone vanity counter / architectural floor with subtle skylight specular bounce
      float fresnelFloor = pow(1.0 - max(0.0, -rd.y), 3.0);
      col = mix(floorCol, skyCol * 0.42, fresnelFloor * 0.30);
    } else if (isWindowWall > 0.5) {
      // Left frosted clerestory window casting soft luminous sheer-draped daylight into the room
      float winSdf = roundedBoxSdf(hitPos.zy - vec2(-1.15, 0.24), vec2(0.75, 0.60), 0.04);
      float winMask = 1.0 - smoothstep(-0.03, 0.12, winSdf);
      float louvers = 0.92 + 0.08 * sin(hitPos.z * 26.0 + sin(uTime * 0.3) * 0.2);
      col = mix(wallCol, skyCol * louvers, winMask * 0.88);
    } else {
      // Back wall with recessed architectural reveal seam & directional skylight bounce
      float seamDist = abs(hitPos.x + 0.28);
      float seamLine = exp(-seamDist * 42.0);
      float seamRightDist = abs(hitPos.x - 0.56);
      float seamRightLine = exp(-seamRightDist * 46.0);
      float wallVerticalAO = smoothstep(-0.82, 0.95, hitPos.y);
      col *= mix(0.66, 1.08, wallVerticalAO);
      col = mix(col, col * 0.68, max(seamLine, seamRightLine * 0.8) * 0.42);

      // Soft warm/cool daylight bounce from ceiling cove & left window across upper back wall
      float skyWash = exp(-pow(max(0.0, 1.05 - hitPos.y), 2.0) * 2.2);
      float leftBounce = smoothstep(0.4, -1.5, hitPos.x);
      col += skyCol * (skyWash * 0.16 + leftBounce * 0.08) * (1.0 + uAudioEnergy * 0.14);
    }

    // Smooth ambient occlusion at 3D room corners & wall-floor creases
    float edgeDistX = min(abs(hitPos.x - 1.75), abs(hitPos.x + 1.55));
    float edgeDistY = min(abs(hitPos.y - 1.05), abs(hitPos.y + 0.82));
    float edgeDistZ = abs(hitPos.z + 2.25);
    float cornerAO = smoothstep(0.0, 0.45, min(edgeDistX, edgeDistY) + edgeDistZ * 0.4);
    col *= mix(0.74, 1.0, cornerAO);

    // Subtle physical low-iron mirror glass tint
    col *= vec3(0.985, 1.0, 0.995);

    // True normalized 3D Z-depth from glass surface for volumetric DOF fog blur
    float distFromGlass = clamp((tHit - 0.65) / 2.65, 0.06, 1.0);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), distFromGlass);
  }
`;

const MEDIA_QUAD_FRAGMENT = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTex;
  uniform float uScreenAspect;
  uniform float uMediaAspect;
  uniform float uMirrorFlip;

  void main() {
    vec2 uv = vec2(mix(vUv.x, 1.0 - vUv.x, uMirrorFlip), vUv.y);
    float ratio = uScreenAspect / max(0.2, uMediaAspect);
    if (ratio > 1.0) {
      uv.y = (uv.y - 0.5) / ratio + 0.5;
    } else {
      uv.x = (uv.x - 0.5) * ratio + 0.5;
    }

    vec3 col = texture2D(uTex, clamp(uv, 0.001, 0.999)).rgb;
    col *= vec3(0.985, 1.0, 0.995);
    // Gentle perspective depth gradient for uploaded video/camera feeds (foreground center closer, perimeter deeper)
    float radialDepth = clamp(length(vUv - 0.5) * 1.15, 0.18, 0.85);
    gl_FragColor = vec4(col, radialDepth);
  }
`;

export function createReflectionWorld(_isSoftwareGl: boolean): ReflectionWorld {
  const studioScene = new THREE.Scene();
  const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quadGeo = new THREE.PlaneGeometry(2, 2);

  const studioMat = new THREE.ShaderMaterial({
    uniforms: {
      uAspect: { value: 1.0 },
      uTime: { value: 0 },
      uParallaxX: { value: 0 },
      uParallaxY: { value: 0 },
      uAudioEnergy: { value: 0 },
      uTheme: { value: 0.0 },
    },
    vertexShader: QUAD_VERTEX,
    fragmentShader: ARCHITECTURAL_REFLECTION_FRAGMENT,
    depthTest: false,
    depthWrite: false,
  });
  const studioQuad = new THREE.Mesh(quadGeo, studioMat);
  studioScene.add(studioQuad);

  const mediaScene = new THREE.Scene();
  const mediaQuadMat = new THREE.ShaderMaterial({
    uniforms: {
      uTex: { value: null },
      uScreenAspect: { value: 1 },
      uMediaAspect: { value: 16 / 9 },
      uMirrorFlip: { value: 1.0 },
    },
    vertexShader: QUAD_VERTEX,
    fragmentShader: MEDIA_QUAD_FRAGMENT,
    depthTest: false,
    depthWrite: false,
  });
  const mediaQuad = new THREE.Mesh(quadGeo, mediaQuadMat);
  mediaScene.add(mediaQuad);

  let videoTexture: THREE.VideoTexture | null = null;
  let activeVideoEl: HTMLVideoElement | null = null;

  let customTex: THREE.Texture | null = null;
  let customMediaEl: HTMLImageElement | HTMLVideoElement | null = null;

  return {
    setCameraStream: (video: HTMLVideoElement | null) => {
      if (videoTexture) {
        videoTexture.dispose();
        videoTexture = null;
      }
      activeVideoEl = video;
      if (video) {
        videoTexture = new THREE.VideoTexture(video);
        videoTexture.colorSpace = THREE.SRGBColorSpace;
        videoTexture.minFilter = THREE.LinearFilter;
        videoTexture.magFilter = THREE.LinearFilter;
      }
    },
    setCustomMedia: (media: HTMLImageElement | HTMLVideoElement | null) => {
      if (customTex) {
        customTex.dispose();
        customTex = null;
      }
      customMediaEl = media;
      if (media instanceof HTMLVideoElement) {
        const tex = new THREE.VideoTexture(media);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        customTex = tex;
      } else if (media instanceof HTMLImageElement) {
        const tex = new THREE.Texture(media);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.needsUpdate = true;
        customTex = tex;
      }
    },
    renderSharpReflection: (
      renderer,
      target,
      aspect,
      elapsed,
      audioEnergy,
      parallaxX,
      parallaxY,
      mode,
      studioTheme,
      dt = 0.016,
    ) => {
      // Real frame dt keeps damping speed identical on 30/60/120Hz displays
      const safeDt = Math.min(0.1, Math.max(0.001, dt));
      const prevTarget = renderer.getRenderTarget();
      renderer.setRenderTarget(target);
      renderer.clear();

      if (
        mode === 'camera' &&
        videoTexture &&
        activeVideoEl &&
        activeVideoEl.readyState >= 2
      ) {
        const vw = activeVideoEl.videoWidth || 1280;
        const vh = activeVideoEl.videoHeight || 720;
        mediaQuadMat.uniforms.uTex.value = videoTexture;
        mediaQuadMat.uniforms.uScreenAspect.value = aspect;
        mediaQuadMat.uniforms.uMediaAspect.value = vw / Math.max(1, vh);
        mediaQuadMat.uniforms.uMirrorFlip.value = 1.0;
        renderer.render(mediaScene, orthoCam);
      } else if (mode === 'media' && customTex && customMediaEl) {
        if (customMediaEl instanceof HTMLVideoElement) {
          if (customMediaEl.readyState >= 2) {
            const mw = customMediaEl.videoWidth || 1280;
            const mh = customMediaEl.videoHeight || 720;
            mediaQuadMat.uniforms.uTex.value = customTex;
            mediaQuadMat.uniforms.uScreenAspect.value = aspect;
            mediaQuadMat.uniforms.uMediaAspect.value = mw / Math.max(1, mh);
            mediaQuadMat.uniforms.uMirrorFlip.value = 0.0;
            renderer.render(mediaScene, orthoCam);
          } else {
            studioMat.uniforms.uAspect.value = aspect;
            studioMat.uniforms.uTime.value = elapsed;
            studioMat.uniforms.uTheme.value = studioTheme === 'velvet_slate' ? 1.0 : 0.0;
            renderer.render(studioScene, orthoCam);
          }
        } else if (customMediaEl.complete) {
          const mw = customMediaEl.naturalWidth || 1280;
          const mh = customMediaEl.naturalHeight || 720;
          mediaQuadMat.uniforms.uTex.value = customTex;
          mediaQuadMat.uniforms.uScreenAspect.value = aspect;
          mediaQuadMat.uniforms.uMediaAspect.value = mw / Math.max(1, mh);
          mediaQuadMat.uniforms.uMirrorFlip.value = 0.0;
          renderer.render(mediaScene, orthoCam);
        }
      } else {
        studioMat.uniforms.uAspect.value = aspect;
        studioMat.uniforms.uTime.value = elapsed;
        studioMat.uniforms.uParallaxX.value = THREE.MathUtils.damp(
          studioMat.uniforms.uParallaxX.value,
          parallaxX,
          6,
          safeDt,
        );
        studioMat.uniforms.uParallaxY.value = THREE.MathUtils.damp(
          studioMat.uniforms.uParallaxY.value,
          parallaxY,
          6,
          safeDt,
        );
        studioMat.uniforms.uAudioEnergy.value = audioEnergy;
        const targetTheme = studioTheme === 'velvet_slate' ? 1.0 : 0.0;
        studioMat.uniforms.uTheme.value = THREE.MathUtils.damp(
          studioMat.uniforms.uTheme.value,
          targetTheme,
          8,
          safeDt,
        );
        renderer.render(studioScene, orthoCam);
      }

      renderer.setRenderTarget(prevTarget);
    },
    dispose: () => {
      if (videoTexture) videoTexture.dispose();
      if (customTex) customTex.dispose();
      quadGeo.dispose();
      studioMat.dispose();
      mediaQuadMat.dispose();
    },
  };
}

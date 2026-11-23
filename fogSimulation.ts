import * as THREE from 'three';
import type { FogPhase, FogStatus } from './types';
import {
  getTrueStrokeOrderGlyph,
  preloadStrokeMediansForText,
  type TrueStrokeGlyph,
} from './strokeOrderData';

const WIPE_MAX_DIM = 1920;
const DROPLET_MAX_DIM = 1440;
const WET_FILM_MAX_DIM = 960;
const WET_GRID_SIZE = 96;
const SKEL_GRID = 148;

// 8-bit canvases cannot fade with tiny alpha values: v * (1 - 0.011) rounds back to v for
// every v below ~45, so old wipe trails never fully heal. Batch the fade into fewer, larger
// steps (same average speed) so values reach 0. Set to false to restore the old behaviour.
const FADE_FLOOR_FIX = true;
const MIN_WIPE_FADE_ALPHA = 0.04;
const MIN_WET_FADE_ALPHA = 0.05;

const QUAD_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const BLUR_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTex;
  uniform vec2 uTexelOffset;

  void main() {
    vec4 centerSample = texture2D(uTex, vUv);
    vec3 sum = centerSample.rgb * 0.2270270270;
    sum += texture2D(uTex, clamp(vUv + uTexelOffset * 1.3846153846, 0.001, 0.999)).rgb * 0.3162162162;
    sum += texture2D(uTex, clamp(vUv - uTexelOffset * 1.3846153846, 0.001, 0.999)).rgb * 0.3162162162;
    sum += texture2D(uTex, clamp(vUv + uTexelOffset * 3.2307692308, 0.001, 0.999)).rgb * 0.0702702703;
    sum += texture2D(uTex, clamp(vUv - uTexelOffset * 3.2307692308, 0.001, 0.999)).rgb * 0.0702702703;
    gl_FragColor = vec4(sum, centerSample.a);
  }
`;

// Dedicated Liquid Mechanics & High-Precision Tangent Normal Map Pass
// Computes 3x3 Sobel meniscus normals + gravity-slumped droplet dome profiles + fluid thickness
export const FLUID_NORMAL_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;

  uniform sampler2D uWipeMap;
  uniform sampler2D uDropletMap;
  uniform sampler2D uWetFilmMap;
  uniform vec2 uWipeTexSize;
  uniform vec2 uDropTexSize;
  uniform vec2 uWetTexSize;
  uniform vec2 uGravityVec;
  uniform float uTime;

  float sampleFluidHeight(vec2 uv, out float dropH, out float filmH, out float wipeVal, out float pileVal) {
    vec2 gDir = normalize(uGravityVec + vec2(0.0001, -0.0001));
    // Asymmetric gravity sag shifts upper droplet height toward bottom belly
    float d0 = texture2D(uDropletMap, uv).r;
    float dUp = texture2D(uDropletMap, clamp(uv - gDir * (1.8 / max(vec2(64.0), uDropTexSize)), 0.001, 0.999)).r;
    dropH = d0 * 0.74 + dUp * 0.26;

    vec4 wet = texture2D(uWetFilmMap, uv);
    filmH = wet.r;
    pileVal = wet.g;
    wipeVal = texture2D(uWipeMap, uv).r;

    // Capillary concave groove along finger path + raised liquid rim around wiped channel
    float edgeRim = smoothstep(0.08, 0.55, wipeVal) * (1.0 - smoothstep(0.60, 0.96, wipeVal));
    float waterChannel = wipeVal * 0.26 + edgeRim * 0.34 + filmH * 0.42;
    return dropH * 1.15 + waterChannel;
  }

  void main() {
    vec2 dTex = vec2(1.15) / max(vec2(64.0), uDropTexSize);
    vec2 wTex = vec2(1.6) / max(vec2(64.0), uWetTexSize);
    vec2 stepUv = mix(wTex, dTex, 0.55);

    float dropC, filmC, wipeC, pileC;
    float hC = sampleFluidHeight(vUv, dropC, filmC, wipeC, pileC);

    float dDummy, fDummy, wDummy, pDummy;
    // 8-neighbor Sobel heightfield stencil for smooth high-precision liquid normals
    float hTL = sampleFluidHeight(clamp(vUv + vec2(-stepUv.x,  stepUv.y), 0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hTC = sampleFluidHeight(clamp(vUv + vec2(0.0,       stepUv.y), 0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hTR = sampleFluidHeight(clamp(vUv + vec2( stepUv.x,  stepUv.y), 0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hML = sampleFluidHeight(clamp(vUv + vec2(-stepUv.x,  0.0),      0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hMR = sampleFluidHeight(clamp(vUv + vec2( stepUv.x,  0.0),      0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hBL = sampleFluidHeight(clamp(vUv + vec2(-stepUv.x, -stepUv.y), 0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hBC = sampleFluidHeight(clamp(vUv + vec2(0.0,      -stepUv.y), 0.001, 0.999), dDummy, fDummy, wDummy, pDummy);
    float hBR = sampleFluidHeight(clamp(vUv + vec2( stepUv.x, -stepUv.y), 0.001, 0.999), dDummy, fDummy, wDummy, pDummy);

    float gx = (hTR + 2.0 * hMR + hBR) - (hTL + 2.0 * hML + hBL);
    float gy = (hTL + 2.0 * hTC + hTR) - (hBL + 2.0 * hBC + hBR);

    // Subtle capillary surface micro-ripple along gravity direction in pooled trails
    float microWave = sin((vUv.y - vUv.x * 0.2) * 180.0 - uTime * 2.2) * 0.015 * filmC;
    vec2 slopeXY = vec2(-gx, -gy) * 0.95 + vec2(0.0, microWave);
    vec3 N = normalize(vec3(slopeXY, 0.52));

    float thickness = clamp(dropC * 0.78 + filmC * 0.34 + wipeC * 0.18 + hC * 0.05, 0.0, 1.0);
    float mistAccum = clamp(pileC, 0.0, 1.0);

    gl_FragColor = vec4(N.x * 0.5 + 0.5, N.y * 0.5 + 0.5, thickness, mistAccum);
  }
`;

// Dedicated Stroke Indentation Depth Buffer & Curvature Tension Pass
// Computes: R = signed tactile indentation depth (recessed groove vs pushed fog lip),
// G/B = fog-wall slope gradient XY for directional self-shadow & bevel catchlight,
// A = discrete mean surface curvature κ across droplets and wiped water meniscus edges.
export const STROKE_DEPTH_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;

  uniform sampler2D uWipeMap;
  uniform sampler2D uDropletMap;
  uniform sampler2D uWetFilmMap;
  uniform vec2 uWipeTexSize;
  uniform vec2 uDropTexSize;
  uniform vec2 uWetTexSize;

  float sampleIndentSurface(vec2 uv, out float wVal, out float dVal, out float curvatureField) {
    wVal = texture2D(uWipeMap, uv).r;
    dVal = texture2D(uDropletMap, uv).r;
    vec4 wet = texture2D(uWetFilmMap, uv);
    float film = wet.r;
    float pile = wet.g;
    // Recessed channel (-wipe) vs displaced frosted fog lip (+pile along edge)
    float outerLip = (1.0 - smoothstep(0.18, 0.62, wVal));
    curvatureField = dVal * 1.35 + film * 0.45 + smoothstep(0.06, 0.55, wVal) * outerLip * 0.52;
    return clamp(0.5 + pile * 0.46 * outerLip - wVal * 0.48, 0.0, 1.0);
  }

  void main() {
    vec2 wTex = vec2(2.2) / max(vec2(64.0), uWipeTexSize);
    vec2 dTex = vec2(1.4) / max(vec2(64.0), uDropTexSize);

    float wC, dC, cC;
    float zC = sampleIndentSurface(vUv, wC, dC, cC);

    float wd, dd;
    float cTL, cTC, cTR, cML, cMR, cBL, cBC, cBR;
    float zTL = sampleIndentSurface(clamp(vUv + vec2(-wTex.x,  wTex.y), 0.001, 0.999), wd, dd, cTL);
    float zTC = sampleIndentSurface(clamp(vUv + vec2(0.0,      wTex.y), 0.001, 0.999), wd, dd, cTC);
    float zTR = sampleIndentSurface(clamp(vUv + vec2( wTex.x,  wTex.y), 0.001, 0.999), wd, dd, cTR);
    float zML = sampleIndentSurface(clamp(vUv + vec2(-wTex.x,  0.0),    0.001, 0.999), wd, dd, cML);
    float zMR = sampleIndentSurface(clamp(vUv + vec2( wTex.x,  0.0),    0.001, 0.999), wd, dd, cMR);
    float zBL = sampleIndentSurface(clamp(vUv + vec2(-wTex.x, -wTex.y), 0.001, 0.999), wd, dd, cBL);
    float zBC = sampleIndentSurface(clamp(vUv + vec2(0.0,     -wTex.y), 0.001, 0.999), wd, dd, cBC);
    float zBR = sampleIndentSurface(clamp(vUv + vec2( wTex.x, -wTex.y), 0.001, 0.999), wd, dd, cBR);

    float gx = (zTR + 2.0 * zMR + zBR) - (zTL + 2.0 * zML + zBL);
    float gy = (zTL + 2.0 * zTC + zTR) - (zBL + 2.0 * zBC + zBR);
    vec2 wallGrad = clamp(vec2(-gx, -gy) * 1.45, -1.0, 1.0);

    // High-resolution curvature Laplacian on droplets & meniscus water ridges
    float dN = texture2D(uDropletMap, clamp(vUv + vec2(0.0, dTex.y), 0.001, 0.999)).r;
    float dS = texture2D(uDropletMap, clamp(vUv - vec2(0.0, dTex.y), 0.001, 0.999)).r;
    float dE = texture2D(uDropletMap, clamp(vUv + vec2(dTex.x, 0.0), 0.001, 0.999)).r;
    float dW = texture2D(uDropletMap, clamp(vUv - vec2(dTex.x, 0.0), 0.001, 0.999)).r;
    float dropLap = abs((dN + dS + dE + dW) * 0.25 - dC);
    float meniscusLap = abs((cTL + cTC + cTR + cML + cMR + cBL + cBC + cBR) * 0.125 - cC);
    float curvature = clamp(dropLap * 8.5 + meniscusLap * 3.8 + dC * (1.0 - smoothstep(0.18, 0.65, dC)) * 0.55, 0.0, 1.0);

    gl_FragColor = vec4(zC, wallGrad.x * 0.5 + 0.5, wallGrad.y * 0.5 + 0.5, curvature);
  }
`;

const FROSTED_MIRROR_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  varying vec2 vUv;

  uniform sampler2D uSharpTex;
  uniform sampler2D uNearBlurTex;
  uniform sampler2D uBlurTex;
  uniform sampler2D uWipeMap;
  uniform sampler2D uDropletMap;
  uniform sampler2D uWetFilmMap;
  uniform sampler2D uFluidLayerTex;
  uniform sampler2D uStrokeDepthTex;
  uniform vec2 uWipeTexSize;
  uniform vec2 uDropTexSize;
  uniform vec2 uWetTexSize;
  uniform float uTime;
  uniform float uFogLevel;
  uniform float uCoverage;      // 0.22 (edge only) .. 1.0 (full screen)
  uniform float uTempDiff;      // 0.2 .. 1.0 (thermal differential ΔT)
  uniform float uMirrorAspect;  // viewport width / height
  uniform float uAudioEnergy;   // 0 .. 1 musical bass/vocal energy
  uniform float uBeatPulse;     // 0 .. 1 transient beat breath
  uniform vec2 uParallax;       // viewer tilt parallax for volumetric steam & Fresnel optics

  float hash21(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
  }

  vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
      v += amp * noise(p);
      p = p * 2.03 + vec2(1.7, 9.2);
      amp *= 0.5;
    }
    return v;
  }

  // Ultra-fine sub-millimeter frost grain normal
  vec2 fineFrostNormal(vec2 uv, float scale) {
    vec2 p = uv * scale;
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float minD = 4.0;
    vec2 bestVec = vec2(0.0);
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        vec2 neighbor = vec2(float(x), float(y));
        vec2 rnd = hash22(ip + neighbor);
        vec2 pt = neighbor + 0.15 + 0.70 * rnd - fp;
        float d = dot(pt, pt);
        if (d < minD) {
          minD = d;
          bestVec = pt;
        }
      }
    }
    return bestVec;
  }

  void main() {
    float safeAspect = max(0.3, uMirrorAspect);
    vec2 normUv = vec2(vUv.x * safeAspect, vUv.y) / min(1.0, safeAspect);

    // 2D Thermal Conduction Field: Corners lose heat in X & Y -> coldest & fastest to condense
    vec2 centered = (vUv - 0.5) * 2.0;
    float ax = abs(centered.x);
    float ay = abs(centered.y);
    float edgeProx = max(ax, ay);
    float cornerProx = smoothstep(0.30, 1.0, ax) * smoothstep(0.30, 1.0, ay);
    float radialCorner = clamp(length(centered) / 1.4142, 0.0, 1.0);
    float thermalColdness = clamp(
      edgeProx * 0.54 + cornerProx * 0.36 + radialCorner * 0.18,
      0.0,
      1.0
    );

    // Silky rising vapor convection currents
    float mistBroad = fbm(normUv * 2.6 + vec2(0.0, -uTime * 0.055));
    float mistDetail = fbm(normUv * 9.5 + vec2(uTime * 0.020, -uTime * 0.072));
    float mistMicro = noise(normUv * 46.0 + vec2(1.4, -uTime * 0.036));
    float thermalMask = mistBroad * 0.58 + mistDetail * 0.30 + mistMicro * 0.12;

    // Temperature differential ΔT modulates corner/edge condensation spread
    float tempFactor = clamp(uTempDiff, 0.15, 1.0);
    float innerReach = 1.0 - clamp(uCoverage, 0.15, 1.0);
    float coverageGate = smoothstep(
      innerReach - 0.25,
      innerReach + mix(0.24, 0.12, tempFactor),
      thermalColdness + (thermalMask - 0.5) * 0.24
    );
    if (uCoverage > 0.94) {
      coverageGate = mix(coverageGate, 1.0, smoothstep(0.94, 1.0, uCoverage));
    }

    float cornerAccel = thermalColdness * mix(0.18, 0.52, tempFactor);
    float beatBreath = uBeatPulse * 0.05;
    float threshold = mix(0.66, 0.08, thermalColdness) - cornerAccel * 0.30 - (uFogLevel + beatBreath) * 1.24;
    float rawCondensation = smoothstep(threshold - 0.26, threshold + 0.32, thermalMask);
    float fogDensity = clamp(rawCondensation * smoothstep(0.0, 0.08, uFogLevel) * coverageGate, 0.0, 1.0);

    // Wipe knockout map + Surface Wetness Coefficient (liquid film left behind after fog displacement/evaporation)
    vec2 wTexel = vec2(1.5) / max(vec2(64.0), uWipeTexSize);
    float wipedRaw = texture2D(uWipeMap, vUv).r;
    float wN = texture2D(uWipeMap, vUv + vec2(0.0, wTexel.y)).r;
    float wS = texture2D(uWipeMap, vUv - vec2(0.0, wTexel.y)).r;
    float wE = texture2D(uWipeMap, vUv + vec2(wTexel.x, 0.0)).r;
    float wW = texture2D(uWipeMap, vUv - vec2(wTexel.x, 0.0)).r;
    vec2 wipeSlope = vec2(wW - wE, wS - wN);
    float wipeEdgeGrad = clamp(length(wipeSlope) * 1.2, 0.0, 1.0)
      * (1.0 - smoothstep(0.50, 0.94, wipedRaw));

    vec2 wetTexel = vec2(1.4) / max(vec2(64.0), uWetTexSize);
    vec4 wetSample = texture2D(uWetFilmMap, vUv);
    float wetFilm = wetSample.r;
    float mistPile = wetSample.g;

    // Dedicated Fluid Mechanics Layer: high-precision tangent-space liquid normal + thickness + mist pile-up
    vec4 fluidData = texture2D(uFluidLayerTex, vUv);
    vec2 fluidNormalXY = (fluidData.rg * 2.0 - 1.0);
    float fluidThickness = fluidData.b;
    float fluidMistAccum = max(mistPile, fluidData.a);

    // Stroke Indentation Depth Buffer & Surface-Tension Curvature Map
    vec4 strokeDepthData = texture2D(uStrokeDepthTex, vUv);
    float indentDepth = clamp((0.5 - strokeDepthData.r) * 2.08, 0.0, 1.0);
    float lipRise = clamp((strokeDepthData.r - 0.5) * 2.08, 0.0, 1.0);
    vec2 wallGrad = strokeDepthData.gb * 2.0 - 1.0;
    float surfaceCurvature = strokeDepthData.a;

    vec2 fN = vec2(
      texture2D(uWetFilmMap, vUv + vec2(wetTexel.x, 0.0)).r - texture2D(uWetFilmMap, vUv - vec2(wetTexel.x, 0.0)).r,
      texture2D(uWetFilmMap, vUv + vec2(0.0, wetTexel.y)).r - texture2D(uWetFilmMap, vUv - vec2(0.0, wetTexel.y)).r
    );
    vec2 filmSlope = mix(-fN, fluidNormalXY, 0.65);

    // Surface Wetness Coefficient: persists as a thin refractive water film even as mist recedes or heals
    float surfaceWetness = clamp(
      wetFilm + max(0.0, wipedRaw - 0.15) * 0.30 + fluidThickness * 0.24 + (1.0 - smoothstep(0.15, 0.65, uFogLevel)) * 0.14 * thermalColdness,
      0.0,
      1.0
    );

    float wiped = clamp(wipedRaw - thermalColdness * tempFactor * 0.035 * (1.0 - wipedRaw), 0.0, 1.0);
    // Displaced fog particles accumulate strictly along outer borders without bleeding inside wiped strokes
    float outerBorderMask = (1.0 - smoothstep(0.14, 0.50, wipedRaw));
    float piledFogDensity = clamp(fogDensity + max(fluidMistAccum, lipRise * 0.65) * 0.24 * outerBorderMask, 0.0, 1.0);
    float effectiveFog = clamp(piledFogDensity * (1.0 - wiped * 0.998), 0.0, 1.0);

    // Ultra-fine physical condensation micro-refraction (soft velvety mist, zero pebble bubbles)
    float grainFreq = mix(280.0, 340.0, thermalColdness * tempFactor);
    vec2 microNormal = fineFrostNormal(normUv, grainFreq);
    float refractAmt = 0.0026 * smoothstep(0.04, 0.65, effectiveFog);
    vec2 distUv = clamp(vUv + microNormal * refractAmt, 0.002, 0.998);

    // 3D Snell Refraction + Depth-Buffered Recessed Trench Parallax Offset
    vec3 viewVec = normalize(vec3(-uParallax.x * 0.68, -uParallax.y * 0.68, 1.0));
    vec3 fluidNormal3D = normalize(vec3(fluidNormalXY * (1.15 + fluidThickness * 0.95), 1.0));
    vec3 snellRefractDir = refract(-viewVec, fluidNormal3D, 0.75);
    vec2 trenchParallax = -uParallax * (0.0038 * indentDepth * fogDensity);
    vec2 filmWarpUv = clamp(
      vUv + trenchParallax + snellRefractDir.xy * (0.0085 + fluidThickness * 0.018) * surfaceWetness,
      0.002,
      0.998
    );
    vec4 sharpSample = texture2D(uSharpTex, filmWarpUv);
    float distFromGlass = sharpSample.a;
    vec3 sharpMirror = sharpSample.rgb;
    vec3 nearBlur = texture2D(uNearBlurTex, distUv).rgb;
    vec3 farBlur = texture2D(uBlurTex, distUv).rgb;

    float clearDof = smoothstep(0.28, 0.98, distFromGlass) * 0.48;
    vec3 clearDofMirror = mix(sharpMirror, nearBlur, clearDof * 0.72);
    clearDofMirror = mix(clearDofMirror, farBlur, smoothstep(0.65, 1.0, distFromGlass) * 0.35);

    // Curvature-Driven Water-Film Specular & Tilt-Responsive Fresnel Reflection
    vec2 filmMicroTex = vec2(
      noise(normUv * 42.0) - 0.5,
      noise(normUv * 42.0 + vec2(8.3, 4.7)) - 0.5
    ) * (0.08 * surfaceWetness);
    vec3 filmNormal = normalize(vec3(fluidNormalXY * 1.65 + filmSlope * 1.4 + wallGrad * 0.12 + filmMicroTex, 1.0));
    float ndotv = clamp(dot(filmNormal, viewVec), 0.0, 1.0);
    float fresnelWater = 0.0204 + (1.0 - 0.0204) * pow(1.0 - ndotv, 3.6);

    vec3 tiltLightDir = normalize(vec3(-0.34 + uParallax.x * 0.42, 0.74 + uParallax.y * 0.42, 0.58));
    vec3 filmHalf = normalize(tiltLightDir + viewVec);
    float curvExp = mix(42.0, 145.0, surfaceCurvature);
    float curvSpecBoost = 0.32 + surfaceCurvature * 1.25;
    float filmSpec = pow(max(dot(filmNormal, filmHalf), 0.0), curvExp) * curvSpecBoost * smoothstep(0.04, 0.55, surfaceWetness);
    float filmRidgeGlint = smoothstep(0.04, 0.36, length(fluidNormalXY) + surfaceCurvature * 0.45) * (0.35 + 0.65 * fresnelWater);

    // Apply wet-film gloss, curvature-focused edge specular highlights & tactile trench wall occlusion
    vec3 envGloss = mix(vec3(0.76, 0.82, 0.88), vec3(0.96, 0.98, 1.0), clamp(uParallax.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 wetGlassMirror = mix(clearDofMirror, envGloss, clamp(fresnelWater * 0.40 * surfaceWetness, 0.0, 0.24));
    wetGlassMirror += vec3(0.95, 0.98, 1.0) * (filmSpec * 0.30 + filmRidgeGlint * 0.16);

    // 3D Carved Trench Wall Contact Shadow & Lip Catchlight (makes handwriting furrows feel physically embossed into fog)
    vec2 lightXY = normalize(vec2(-0.34 + uParallax.x * 0.3, 0.94));
    float indentShadowWall = max(0.0, dot(wallGrad, lightXY)) * smoothstep(0.04, 0.45, length(wallGrad)) * fogDensity;
    float indentSunlitLip = max(0.0, dot(wallGrad, -lightXY)) * smoothstep(0.04, 0.45, length(wallGrad)) * fogDensity;
    wetGlassMirror *= (1.0 - indentShadowWall * 0.04);
    wetGlassMirror += vec3(0.90, 0.95, 1.0) * (indentSunlitLip * 0.03);

    float volumetricCoc = clamp(
      smoothstep(0.06, 0.92, distFromGlass) * (0.48 + 0.52 * effectiveFog) + effectiveFog * 0.45,
      0.0,
      1.0
    );
    vec3 dofSteamedRef = mix(sharpMirror, nearBlur, smoothstep(0.04, 0.48, volumetricCoc));
    dofSteamedRef = mix(dofSteamedRef, farBlur, smoothstep(0.32, 0.95, volumetricCoc));

    // Volumetric 3-Slice Atmospheric Parallax Steam Strata
    vec2 midVaporUv = normUv + uParallax * 0.014;
    vec2 farVaporUv = normUv + uParallax * 0.030;
    float nearVaporSlice = fbm(midVaporUv * 2.1 + vec2(uTime * 0.018, -uTime * 0.11));
    float farVaporSlice = fbm(farVaporUv * 1.45 + vec2(-uTime * 0.014, -uTime * 0.14));
    float volumetricPlume = clamp(nearVaporSlice * 0.58 + farVaporSlice * 0.42, 0.0, 1.0);

    float microShimmer = noise(normUv * 195.0) * 0.022 + noise(normUv * 78.0) * 0.016;
    vec2 lightDir = normalize(vec2(-0.34, 0.88));
    float glint = pow(max(dot(normalize(microNormal + 0.001), lightDir), 0.0), 10.0) * 0.024;

    vec3 warmVaporCol = vec3(0.89, 0.91, 0.93);
    vec3 coldCornerFrostCol = vec3(0.82, 0.87, 0.93);
    vec3 depthHazeCol = vec3(0.86, 0.90, 0.94);
    vec3 vaporScatterCol = mix(warmVaporCol, coldCornerFrostCol, thermalColdness * (0.42 + 0.58 * tempFactor));

    float depthScatter = 1.0 - exp(-distFromGlass * (0.36 + 1.05 * effectiveFog));
    float hazeWeight = clamp(0.42 + 0.36 * effectiveFog + depthScatter * 0.14 + microShimmer + mistPile * 0.06, 0.0, 0.86);
    vec3 steamedGlass = mix(dofSteamedRef, vaporScatterCol, hazeWeight);
    steamedGlass += vec3(0.95, 0.97, 1.0) * (glint + uAudioEnergy * 0.02 + mistPile * 0.03) * effectiveFog;

    // Razor-crisp wet-glass reveal wherever fingers wiped, lyrics were drawn, or water rivulets swept clean
    float fogRevealMask = smoothstep(0.02, 0.44, effectiveFog);
    vec3 finalColor = mix(wetGlassMirror, steamedGlass, fogRevealMask);

    // Floating volumetric room steam drift
    float airborneSteam = smoothstep(0.38, 0.88, volumetricPlume)
      * clamp(uFogLevel * 0.15 + uBeatPulse * 0.06, 0.0, 0.22)
      * (1.0 - wiped * 0.88)
      * smoothstep(0.08, 0.85, distFromGlass + 0.25);
    finalColor = mix(finalColor, depthHazeCol, airborneSteam);

    // Tactile fog-ridge emboss highlight + inner trench ambient occlusion along mist rim
    float rimCatch = max(dot(normalize(wipeSlope + vec2(0.0001)), lightDir), 0.0);
    finalColor *= (1.0 - indentShadowWall * 0.03 * fogRevealMask);
    finalColor += vec3(0.038, 0.046, 0.054) * (wipeEdgeGrad * (0.20 + 0.30 * rimCatch) + indentSunlitLip * 0.15) * fogDensity;
    // Cleared strokes keep a faint milky veil so they read as wet glass, not black ink
    finalColor = mix(finalColor, vaporScatterCol, 0.16 * fogDensity * (1.0 - fogRevealMask));

    // Photorealistic 3D Inverted-Lens Water Droplets with Curvature-Based Meniscus Specular Highlights
    float dCenter = texture2D(uDropletMap, vUv).r;
    if (dCenter > 0.015) {
      vec3 surfNormal = normalize(vec3(fluidNormalXY * (2.55 + fluidThickness * 1.15), max(0.22, dCenter * 0.94)));
      vec3 halfVec = normalize(tiltLightDir + viewVec);
      vec3 dropRefract = refract(-viewVec, surfNormal, 0.75);

      vec2 lensUv = clamp(vUv + dropRefract.xy * (0.046 + fluidThickness * 0.042), 0.002, 0.998);
      vec3 refractedDrop = texture2D(uSharpTex, lensUv).rgb;

      float dropMask = smoothstep(0.04, 0.18, dCenter);
      float rimBand = smoothstep(0.04, 0.22, dCenter) * (1.0 - smoothstep(0.28, 0.56, dCenter));
      float fresnelDrop = 0.0204 + (1.0 - 0.0204) * pow(1.0 - max(dot(surfNormal, viewVec), 0.0), 2.6);

      // Gentle upper-cap refraction shadow + radiant lower meniscus caustic crescent + curvature meniscus gleam
      float upperCapDark = smoothstep(0.08, 0.72, surfNormal.y);
      float lowerBellyGlow = pow(max(0.0, dot(surfNormal.xy, normalize(vec2(0.08, -0.99)))), 1.55)
        * smoothstep(0.10, 0.60, dCenter);
      float dropShininess = mix(72.0, 165.0, surfaceCurvature);
      float specHighlight = pow(max(dot(surfNormal, halfVec), 0.0), dropShininess) * (0.85 + surfaceCurvature * 0.65) * dropMask;
      float edgeTensionGlint = rimBand * fresnelDrop * (0.45 + surfaceCurvature * 1.35);
      vec3 glassRimEnv = mix(vec3(0.76, 0.83, 0.90), vec3(0.97, 0.99, 1.0), clamp(-surfNormal.y * 0.5 + 0.5, 0.0, 1.0));

      vec3 dropBody = mix(refractedDrop * 1.12, glassRimEnv, clamp(fresnelDrop * (0.44 + surfaceCurvature * 0.24), 0.0, 0.78));
      dropBody *= (1.0 - upperCapDark * 0.26 - rimBand * 0.20);
      dropBody += mix(vec3(0.86, 0.92, 0.97), vec3(1.0), 0.68) * (lowerBellyGlow * 0.62);
      dropBody += vec3(0.99, 0.995, 1.0) * (specHighlight * 0.94 + edgeTensionGlint * 0.48);

      finalColor = mix(finalColor, dropBody, dropMask * 0.93);
    }

    gl_FragColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);
  }
`;

function computeAspectDims(aspect: number, maxDim: number): { w: number; h: number } {
  const safeAspect = Math.max(0.32, Math.min(3.2, aspect));
  if (safeAspect >= 1) {
    return {
      w: maxDim,
      h: Math.max(480, Math.round(maxDim / safeAspect)),
    };
  }
  return {
    w: Math.max(420, Math.round(maxDim * safeAspect)),
    h: maxDim,
  };
}

// Maps user strokeWeight (0.15..1.60) to crisp finger wipe radius in normalized screen heights
function strokeWeightToFingerRadius(weight: number): number {
  const t = THREE.MathUtils.clamp((weight - 0.15) / (1.6 - 0.15), 0, 1);
  return THREE.MathUtils.lerp(0.0052, 0.046, Math.pow(t, 1.32));
}

function createSteamParticleTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const grad = ctx.createRadialGradient(
    size / 2,
    size / 2,
    4,
    size / 2,
    size / 2,
    size / 2,
  );
  grad.addColorStop(0, 'rgba(240, 245, 250, 0.22)');
  grad.addColorStop(0.45, 'rgba(226, 234, 242, 0.08)');
  grad.addColorStop(1, 'rgba(226, 234, 242, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

interface Pt2 {
  x: number;
  y: number;
}

interface RawStrokeData {
  points: Pt2[];
  fieldDirs: Pt2[];
  isClosed: boolean;
}

interface ExtractedGlyph {
  strokes: RawStrokeData[];
  inkMinX: number;
  inkMaxX: number;
  exitPt: Pt2;
  entryPt: Pt2;
  strokeCount: number;
}

interface HandwritingStroke {
  points: Array<{ u: number; v: number }>;
  fieldDirs: Array<{ u: number; v: number }>;
  cumLengths: number[];
  curvatures: number[];
  totalLength: number;
  isSmallDetail: boolean;
  linkToPrev: 'lift' | 'ligature' | 'flow';
}

interface ActiveHandwritingJob {
  strokes: HandwritingStroke[];
  strokeIndex: number;
  targetDist: number;
  strokeElapsed: number;
  tipU: number;
  tipV: number;
  tipVx: number;
  tipVy: number;
  brushRadius: number;
  strokeStarted: boolean;
  liftCooldownSec: number;
  isLigatureJump: boolean;
  ligatureStartU: number;
  ligatureStartV: number;
  ligatureEndU: number;
  ligatureEndV: number;
  ligatureDurationSec: number;
  ligatureTimer: number;
  tipHoldSec: number;
  baseSpeedUvPerSec: number;
  pressure: number;
  phaseSeed: number;
  wordEndTriggered: boolean;
  finishHoldSec: number;
}

interface EdgeWaterBead {
  u: number;
  v: number;
  prevU: number;
  prevV: number;
  vx: number;
  vy: number;
  r: number;
  life: number;
  maxLife: number;
  seed: number;
  slipPhase: number;
  rolling: boolean;
}

export interface FogSimulationSystem {
  screenMaterial: THREE.ShaderMaterial;
  steamParticles: THREE.Points;
  fluidInputs: {
    wipeMap: THREE.Texture;
    dropletMap: THREE.Texture;
    wetFilmMap: THREE.Texture;
    wipeTexSize: THREE.Vector2;
    dropTexSize: THREE.Vector2;
    wetTexSize: THREE.Vector2;
  };
  setTextures: (
    sharpTex: THREE.Texture,
    nearBlurTex: THREE.Texture,
    farBlurTex: THREE.Texture,
    fluidLayerTex: THREE.Texture,
    strokeDepthTex: THREE.Texture,
  ) => void;
  setMirrorAspect: (aspect: number) => void;
  triggerBeatPulse: (intensity?: number) => void;
  update: (
    dt: number,
    elapsed: number,
    breathCycle: boolean,
    recondenseSpeed: number,
    maxDensity: number,
    fogCoverage: number,
    tempDiff: number,
    audioEnergy?: number,
    lyricsActive?: boolean,
    handwritingScale?: number,
    strokeWeight?: number,
    parallaxX?: number,
    parallaxY?: number,
  ) => FogStatus;
  wipeAtUv: (u: number, v: number, radiusNorm?: number) => void;
  wipeStroke: (
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    dtSec?: number,
  ) => void;
  writeLyricOnGlass: (
    text: string,
    durationSec?: number,
    isChorus?: boolean,
  ) => void;
  triggerSteamBurst: () => void;
  clearAllFog: () => void;
  /** True while wipe / wet-film / droplet maps changed recently (fluid passes need a refresh). */
  isFluidActive: () => boolean;
  dispose: () => void;
}

type ScriptKind = 'latin' | 'cjk' | 'hangul' | 'kana';

function detectScriptKind(text: string): ScriptKind {
  if (/[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/.test(text)) return 'hangul';
  if (/[\u3040-\u309F\u30A0-\u30FF]/.test(text)) return 'kana';
  if (/[\u4E00-\u9FFF]/.test(text)) return 'cjk';
  return 'latin';
}

function resolveHandwritingFont(text: string, fontSize: number): string {
  const kind = detectScriptKind(text);
  if (kind === 'hangul') {
    return `400 ${Math.round(fontSize * 1.12)}px "Gaegu", "Nanum Pen Script", "Klee One", cursive`;
  }
  if (kind === 'kana') {
    return `400 ${Math.round(fontSize * 1.05)}px "Yomogi", "Klee One", "Gaegu", cursive`;
  }
  if (kind === 'cjk') {
    return `400 ${fontSize}px "LXGW WenKai TC", "Klee One", "Kaiti TC", "DFKai-SB", cursive`;
  }
  return `600 ${Math.round(fontSize * 1.24)}px "Caveat", "Gaegu", "Brush Script MT", cursive`;
}

// Tokenize lines into whole Latin words/contractions + individual CJK/Hangul/Kana glyphs
function tokenizeMixedLine(line: string): string[] {
  const matches = line.match(/[A-Za-z0-9'’\-]+|[\u4E00-\u9FFF]|[\uAC00-\uD7AF]|[\u3040-\u309F\u30A0-\u30FF]|\s+|[^\s]/g);
  return matches ?? Array.from(line);
}

function preloadHandwritingFonts() {
  if (typeof document === 'undefined' || !document.fonts?.load) return;
  const samples: Array<[string, string]> = [
    ['400 72px "Gaegu"', '안녕사랑해기억해'],
    ['400 72px "Nanum Pen Script"', '안녕사랑해기억해'],
    ['400 72px "Yomogi"', 'あいうえお君の声が'],
    ['400 72px "Klee One"', '嗨你好的我愛夢忘記鏡子'],
    ['400 72px "LXGW WenKai TC"', '嗨你好的我愛夢忘記鏡子'],
    ['600 72px "Caveat"', 'Hello Love Mirror Dream'],
  ];
  for (const [spec, text] of samples) {
    void document.fonts.load(spec, text).catch(() => {});
  }
}

async function ensureGlyphFontsReady(text: string, fontSize: number): Promise<void> {
  if (typeof document === 'undefined' || !document.fonts?.load) return;
  try {
    const fontsToCheck = new Set<string>();
    for (const tok of tokenizeMixedLine(text)) {
      if (!tok.trim()) continue;
      fontsToCheck.add(resolveHandwritingFont(tok, fontSize));
    }
    if (fontsToCheck.size === 0) fontsToCheck.add(resolveHandwritingFont(text, fontSize));
    await Promise.race([
      Promise.all(Array.from(fontsToCheck).map((f) => document.fonts.load(f, text))),
      new Promise<void>((resolve) => window.setTimeout(resolve, 900)),
    ]);
  } catch {
    // Fallback gracefully
  }
}

function measureMixedLineWidth(
  ctx: CanvasRenderingContext2D,
  tokens: string[],
  fontSize: number,
  cjkTrackingPx: number,
  latinGapPx: number,
): number {
  let sum = 0;
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (!tok.trim()) {
      sum += fontSize * 0.36;
      continue;
    }
    const script = detectScriptKind(tok);
    const chars = Array.from(tok);
    let tokW = 0;
    for (let c = 0; c < chars.length; c++) {
      const ch = chars[c];
      const trueG = getTrueStrokeOrderGlyph(ch);
      if (trueG) {
        tokW += trueG.advance * fontSize;
        if (c < chars.length - 1) tokW += fontSize * 0.03;
      } else {
        ctx.font = resolveHandwritingFont(ch, fontSize);
        tokW += Math.max(ctx.measureText(ch).width, fontSize * 0.45);
      }
    }
    sum += tokW;
    if (i < tokens.length - 1 && tokens[i + 1].trim()) {
      const nextScript = detectScriptKind(tokens[i + 1]);
      if (script === 'latin' && nextScript === 'latin') {
        sum += latinGapPx * 0.25;
      } else if (script !== nextScript) {
        sum += fontSize * 0.16;
      } else {
        sum += cjkTrackingPx * 0.65;
      }
    }
  }
  return sum;
}

function splitBalancedLines(
  ctx: CanvasRenderingContext2D,
  clean: string,
  maxWidth: number,
  fontSize: number,
  cjkTrackingPx: number,
  latinGapPx: number,
): string[] {
  const tokens = tokenizeMixedLine(clean);
  if (measureMixedLineWidth(ctx, tokens, fontSize, cjkTrackingPx, latinGapPx) <= maxWidth) {
    return [clean];
  }

  const hasSpaces = /\s/.test(clean);
  if (hasSpaces) {
    const words = clean.split(/\s+/);
    if (words.length >= 2) {
      let bestIdx = Math.ceil(words.length / 2);
      let bestDiff = Infinity;
      for (let i = 1; i < words.length; i++) {
        const l1 = words.slice(0, i).join(' ');
        const l2 = words.slice(i).join(' ');
        const w1 = measureMixedLineWidth(ctx, tokenizeMixedLine(l1), fontSize, cjkTrackingPx, latinGapPx);
        const w2 = measureMixedLineWidth(ctx, tokenizeMixedLine(l2), fontSize, cjkTrackingPx, latinGapPx);
        const diff = Math.abs(w1 - w2) + (Math.max(w1, w2) > maxWidth ? 2000 : 0);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIdx = i;
        }
      }
      return [
        words.slice(0, bestIdx).join(' '),
        words.slice(bestIdx).join(' '),
      ];
    }
  }

  const chars = Array.from(clean);
  const len = chars.length;
  if (len <= 3) return [clean];

  let bestSplit = Math.ceil(len / 2);
  let bestScore = Infinity;
  const minPart = Math.max(2, Math.floor(len * 0.28));

  for (let i = minPart; i <= len - minPart; i++) {
    const chPrev = chars[i - 1];
    const isPunctBreak = /[，。、！？,.;!?·\s]/.test(chPrev);
    const l1 = chars.slice(0, i).join('');
    const l2 = chars.slice(i).join('');
    const w1 = measureMixedLineWidth(ctx, tokenizeMixedLine(l1), fontSize, cjkTrackingPx, latinGapPx);
    const w2 = measureMixedLineWidth(ctx, tokenizeMixedLine(l2), fontSize, cjkTrackingPx, latinGapPx);
    const overflow = Math.max(0, w1 - maxWidth) + Math.max(0, w2 - maxWidth);
    const balance = Math.abs(w1 - w2);
    const score = overflow * 4 + balance - (isPunctBreak ? maxWidth * 0.35 : 0);
    if (score < bestScore) {
      bestScore = score;
      bestSplit = i;
    }
  }

  const line1 = chars.slice(0, bestSplit).join('').trim();
  const line2 = chars.slice(bestSplit).join('').trim();
  return [line1, line2].filter(Boolean);
}

// Topological Zhang-Suen Thinning + Stroke-Order-Guided Vector Field Extractor
const skeletonCache = new Map<string, ExtractedGlyph>();

function chaikinSmooth(pts: Pt2[], passes = 2, closed = false): Pt2[] {
  let curr = pts;
  for (let p = 0; p < passes; p++) {
    if (curr.length < 3) break;
    const next: Pt2[] = [];
    if (!closed) next.push(curr[0]);
    const end = closed ? curr.length : curr.length - 1;
    for (let i = 0; i < end; i++) {
      const a = curr[i];
      const b = curr[(i + 1) % curr.length];
      next.push({ x: a.x * 0.75 + b.x * 0.25, y: a.y * 0.75 + b.y * 0.25 });
      next.push({ x: a.x * 0.25 + b.x * 0.75, y: a.y * 0.25 + b.y * 0.75 });
    }
    if (!closed) next.push(curr[curr.length - 1]);
    curr = next;
  }
  return curr;
}

function thinAndExtractStrokes(
  grid: Uint8Array,
  W: number,
  H: number,
  spanPx: number,
  cx: number,
  cy: number,
  preferLeftToRight: boolean,
): ExtractedGlyph {
  // 1. Zhang-Suen parallel thinning to 1px skeleton
  let changed = true;
  const toClear: number[] = [];
  let iters = 0;
  while (changed && iters < 30) {
    changed = false;
    iters++;
    for (let step = 0; step < 2; step++) {
      toClear.length = 0;
      for (let y = 1; y < H - 1; y++) {
        const row = y * W;
        for (let x = 1; x < W - 1; x++) {
          const idx = row + x;
          if (!grid[idx]) continue;
          const p2 = grid[(y - 1) * W + x];
          const p3 = grid[(y - 1) * W + x + 1];
          const p4 = grid[y * W + x + 1];
          const p5 = grid[(y + 1) * W + x + 1];
          const p6 = grid[(y + 1) * W + x];
          const p7 = grid[(y + 1) * W + x - 1];
          const p8 = grid[y * W + x - 1];
          const p9 = grid[(y - 1) * W + x - 1];

          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (B < 2 || B > 6) continue;

          let A = 0;
          if (!p2 && p3) A++;
          if (!p3 && p4) A++;
          if (!p4 && p5) A++;
          if (!p5 && p6) A++;
          if (!p6 && p7) A++;
          if (!p7 && p8) A++;
          if (!p8 && p9) A++;
          if (!p9 && p2) A++;
          if (A !== 1) continue;

          if (step === 0) {
            if (p2 * p4 * p6 !== 0) continue;
            if (p4 * p6 * p8 !== 0) continue;
          } else {
            if (p2 * p4 * p8 !== 0) continue;
            if (p2 * p6 * p8 !== 0) continue;
          }
          toClear.push(idx);
        }
      }
      if (toClear.length > 0) {
        changed = true;
        for (let k = 0; k < toClear.length; k++) {
          grid[toClear[k]] = 0;
        }
      }
    }
  }

  const dirs = [
    [-1, -1], [0, -1], [1, -1],
    [-1, 0],           [1, 0],
    [-1, 1],  [0, 1],  [1, 1],
  ];

  const getNeighbors = (idx: number): number[] => {
    const x = idx % W;
    const y = Math.floor(idx / W);
    const out: number[] = [];
    for (const [dx, dy] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx <= 0 || nx >= W - 1 || ny <= 0 || ny >= H - 1) continue;
      const nIdx = ny * W + nx;
      if (grid[nIdx]) out.push(nIdx);
    }
    return out;
  };

  // 2. Topological Spur Pruning: eliminate tiny <=3px serif twigs hanging off junctions
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const idx = y * W + x;
        if (!grid[idx]) continue;
        const nb0 = getNeighbors(idx);
        if (nb0.length !== 1) continue;
        const branch: number[] = [idx];
        let prev = idx;
        let curr = nb0[0];
        while (branch.length <= 3) {
          const nbs = getNeighbors(curr);
          if (nbs.length >= 3) {
            for (const bIdx of branch) grid[bIdx] = 0;
            break;
          }
          const next = nbs.find((n) => n !== prev);
          if (next === undefined) break;
          branch.push(curr);
          prev = curr;
          curr = next;
        }
      }
    }
  }

  const degreeOf = new Uint8Array(W * H);
  const skeletonIndices: number[] = [];
  let pixMinX = W;
  let pixMaxX = 0;
  let pixMinY = H;
  let pixMaxY = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      if (!grid[idx]) continue;
      const nbs = getNeighbors(idx);
      degreeOf[idx] = nbs.length;
      skeletonIndices.push(idx);
      if (x < pixMinX) pixMinX = x;
      if (x > pixMaxX) pixMaxX = x;
      if (y < pixMinY) pixMinY = y;
      if (y > pixMaxY) pixMaxY = y;
    }
  }

  // 3. Build 2D Calligraphic Stroke-Order Vector Field (vfX, vfY)
  // Combines natural spatial stroke-order priors (橫平豎直撇捺 vs Latin left-to-right cursive)
  // with smoothed skeleton structural tangent vectors.
  const vfX = new Float32Array(W * H);
  const vfY = new Float32Array(W * H);
  const boxW = Math.max(1, pixMaxX - pixMinX);
  const boxH = Math.max(1, pixMaxY - pixMinY);

  const calligraphicPrior = (x: number, y: number): Pt2 => {
    const nx = ((x - pixMinX) / boxW) * 2 - 1;
    const ny = ((y - pixMinY) / boxH) * 2 - 1;
    if (preferLeftToRight) {
      // Latin cursive glides right with slight upward slant on entry/exit ascenders
      const px = 0.94;
      const py = ny < -0.15 ? 0.24 : -0.10 + nx * 0.05;
      const len = Math.hypot(px, py) || 1;
      return { x: px / len, y: py / len };
    }
    // CJK/Hangul/Kana: top-left to bottom-right bias, left-side 撇 swings down-left, right-side 捺/勾 swings down-right
    const px = nx < -0.25 && ny > 0.05 ? 0.36 : 0.78;
    const py = 0.74;
    const len = Math.hypot(px, py) || 1;
    return { x: px / len, y: py / len };
  };

  for (const idx of skeletonIndices) {
    const x = idx % W;
    const y = Math.floor(idx / W);
    const prior = calligraphicPrior(x, y);
    const nbs = getNeighbors(idx);
    let tx = prior.x;
    let ty = prior.y;
    if (nbs.length === 2) {
      const a = nbs[0];
      const b = nbs[1];
      let dx = (b % W) - (a % W);
      let dy = Math.floor(b / W) - Math.floor(a / W);
      const dl = Math.hypot(dx, dy) || 1;
      dx /= dl;
      dy /= dl;
      // Align tangent polarity cleanly with calligraphic stroke-order prior
      if (dx * prior.x + dy * prior.y < 0) {
        dx = -dx;
        dy = -dy;
      }
      tx = dx * 0.78 + prior.x * 0.22;
      ty = dy * 0.78 + prior.y * 0.22;
    } else if (nbs.length === 1) {
      const a = nbs[0];
      let dx = (a % W) - x;
      let dy = Math.floor(a / W) - y;
      const dl = Math.hypot(dx, dy) || 1;
      dx /= dl;
      dy /= dl;
      if (dx * prior.x + dy * prior.y < 0) {
        dx = -dx;
        dy = -dy;
      }
      tx = dx * 0.68 + prior.x * 0.32;
      ty = dy * 0.68 + prior.y * 0.32;
    }
    const tl = Math.hypot(tx, ty) || 1;
    vfX[idx] = tx / tl;
    vfY[idx] = ty / tl;
  }

  // Smooth the Vector Field across neighboring pixels so stroke directions stay coherent across junctions
  const vfX2 = new Float32Array(vfX);
  const vfY2 = new Float32Array(vfY);
  for (let pass = 0; pass < 2; pass++) {
    const srcX = pass === 0 ? vfX : vfX2;
    const srcY = pass === 0 ? vfY : vfY2;
    const dstX = pass === 0 ? vfX2 : vfX;
    const dstY = pass === 0 ? vfY2 : vfY;
    for (const idx of skeletonIndices) {
      const nbs = getNeighbors(idx);
      let sx = srcX[idx] * 2.0;
      let sy = srcY[idx] * 2.0;
      for (const n of nbs) {
        const dot = srcX[idx] * srcX[n] + srcY[idx] * srcY[n];
        const w = dot >= 0 ? 1.0 : 0.25;
        sx += srcX[n] * w;
        sy += srcY[n] * w;
      }
      const sl = Math.hypot(sx, sy) || 1;
      dstX[idx] = sx / sl;
      dstY[idx] = sy / sl;
    }
  }

  const sampleVectorField = (nx: number, ny: number): Pt2 => {
    const px = Math.max(1, Math.min(W - 2, Math.round(cx + nx * spanPx)));
    const py = Math.max(1, Math.min(H - 2, Math.round(cy + ny * spanPx)));
    let sx = 0;
    let sy = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const k = (py + dy) * W + (px + dx);
        if (grid[k]) {
          sx += vfX[k];
          sy += vfY[k];
        }
      }
    }
    if (sx === 0 && sy === 0) {
      return calligraphicPrior(px, py);
    }
    const l = Math.hypot(sx, sy) || 1;
    return { x: sx / l, y: sy / l };
  };

  const visitedEdge = new Set<number>();
  const visitedNonJunction = new Uint8Array(W * H);
  const stride = W + 2;
  const edgeKey = (a: number, b: number) => (a < b ? a * stride * H + b : b * stride * H + a);

  const rawPixelStrokes: number[][] = [];

  const traceFrom = (startIdx: number, preferDownRight: boolean) => {
    const firstNeighbors = getNeighbors(startIdx).filter(
      (n) => !visitedEdge.has(edgeKey(startIdx, n)) && !visitedNonJunction[n],
    );
    if (firstNeighbors.length === 0) return;

    const fxStart = vfX[startIdx];
    const fyStart = vfY[startIdx];

    firstNeighbors.sort((a, b) => {
      const ax = (a % W) - (startIdx % W);
      const ay = Math.floor(a / W) - Math.floor(startIdx / W);
      const al = Math.hypot(ax, ay) || 1;
      const bx = (b % W) - (startIdx % W);
      const by = Math.floor(b / W) - Math.floor(startIdx / W);
      const bl = Math.hypot(bx, by) || 1;
      const fieldAlignA = (ax / al) * fxStart + (ay / al) * fyStart;
      const fieldAlignB = (bx / bl) * fxStart + (by / bl) * fyStart;
      const heuristicA = preferLeftToRight
        ? ax * 0.9 + ay * 0.35
        : preferDownRight
          ? ax * 0.65 + ay * 0.95
          : ax - ay;
      const heuristicB = preferLeftToRight
        ? bx * 0.9 + by * 0.35
        : preferDownRight
          ? bx * 0.65 + by * 0.95
          : bx - by;
      return fieldAlignB * 1.45 + heuristicB * 0.55 - (fieldAlignA * 1.45 + heuristicA * 0.55);
    });

    const path: number[] = [startIdx];
    let prev = startIdx;
    let curr = firstNeighbors[0];
    visitedEdge.add(edgeKey(prev, curr));
    if (degreeOf[startIdx] <= 2) visitedNonJunction[startIdx] = 1;
    path.push(curr);

    let dirX = (curr % W) - (prev % W);
    let dirY = Math.floor(curr / W) - Math.floor(prev / W);

    while (path.length < 640) {
      if (degreeOf[curr] <= 2) visitedNonJunction[curr] = 1;
      const candidates = getNeighbors(curr).filter(
        (n) =>
          n !== prev &&
          !visitedEdge.has(edgeKey(curr, n)) &&
          (!visitedNonJunction[n] || n === startIdx),
      );
      if (candidates.length === 0) break;

      const curLen = Math.hypot(dirX, dirY) || 1;
      const ux = dirX / curLen;
      const uy = dirY / curLen;
      const fux = vfX[curr];
      const fuy = vfY[curr];

      let bestNext = -1;
      let bestScore = -2;
      let bestMomentumDot = -2;
      for (const n of candidates) {
        const vx = (n % W) - (curr % W);
        const vy = Math.floor(n / W) - Math.floor(curr / W);
        const vLen = Math.hypot(vx, vy) || 1;
        const momentumDot = (vx / vLen) * ux + (vy / vLen) * uy;
        const fieldDot = (vx / vLen) * fux + (vy / vLen) * fuy;
        const score = momentumDot * 0.78 + fieldDot * 0.22;
        if (score > bestScore) {
          bestScore = score;
          bestMomentumDot = momentumDot;
          bestNext = n;
        }
      }

      const junctionStop = preferLeftToRight ? 0.08 : 0.20;
      if (bestNext < 0 || (degreeOf[curr] >= 3 && bestMomentumDot < junctionStop)) {
        break;
      }

      visitedEdge.add(edgeKey(curr, bestNext));
      prev = curr;
      curr = bestNext;
      path.push(curr);

      const stepDx = (curr % W) - (prev % W);
      const stepDy = Math.floor(curr / W) - Math.floor(prev / W);
      dirX = dirX * 0.48 + stepDx * 0.52;
      dirY = dirY * 0.48 + stepDy * 0.52;

      if (curr === startIdx) break;
    }

    if (path.length >= 2) {
      rawPixelStrokes.push(path);
    }
  };

  // Pass A: endpoints in natural reading/writing sweep weighted by stroke-order vector field
  const endpoints = skeletonIndices
    .filter((idx) => degreeOf[idx] === 1)
    .sort((a, b) => {
      const scoreA = preferLeftToRight
        ? (a % W) * 1.15 + Math.floor(a / W) * 0.25
        : Math.floor(a / W) * 1.18 + (a % W) * 0.82;
      const scoreB = preferLeftToRight
        ? (b % W) * 1.15 + Math.floor(b / W) * 0.25
        : Math.floor(b / W) * 1.18 + (b % W) * 0.82;
      return scoreA - scoreB;
    });
  for (const ep of endpoints) {
    traceFrom(ep, true);
  }

  // Pass B: junctions and closed loops
  const sortedAll = [...skeletonIndices].sort((a, b) => {
    const scoreA = preferLeftToRight
      ? (a % W) * 1.15 + Math.floor(a / W) * 0.25
      : Math.floor(a / W) * 1.18 + (a % W) * 0.82;
    const scoreB = preferLeftToRight
      ? (b % W) * 1.15 + Math.floor(b / W) * 0.25
      : Math.floor(b / W) * 1.18 + (b % W) * 0.82;
    return scoreA - scoreB;
  });

  for (const idx of sortedAll) {
    if (degreeOf[idx] >= 3) {
      let guard = 0;
      while (
        guard++ < 6 &&
        getNeighbors(idx).some(
          (n) => !visitedEdge.has(edgeKey(idx, n)) && !visitedNonJunction[n],
        )
      ) {
        traceFrom(idx, true);
      }
    }
  }
  for (const idx of sortedAll) {
    if (!visitedNonJunction[idx] && degreeOf[idx] >= 1) {
      traceFrom(idx, false);
    }
  }

  // Pass C: Isolated dots ('i', 'j', CJK punctuation dots)
  for (const idx of sortedAll) {
    if (degreeOf[idx] === 0 && !visitedNonJunction[idx]) {
      visitedNonJunction[idx] = 1;
      rawPixelStrokes.push([idx, idx]);
    }
  }

  const strokes: RawStrokeData[] = rawPixelStrokes.map((pixPath) => {
    const isClosed =
      pixPath.length > 5 &&
      pixPath[0] === pixPath[pixPath.length - 1];
    const pts: Pt2[] = [];
    const step = pixPath.length > 10 ? 2 : 1;
    for (let i = 0; i < pixPath.length; i += step) {
      const idx = pixPath[i];
      pts.push({
        x: ((idx % W) - cx) / spanPx,
        y: (Math.floor(idx / W) - cy) / spanPx,
      });
    }
    const lastIdx = pixPath[pixPath.length - 1];
    if (pixPath.length === 2 && pixPath[0] === pixPath[1]) {
      const p0 = pts[0];
      const dotPts = [
        { x: p0.x - 0.015, y: p0.y - 0.015 },
        { x: p0.x + 0.018, y: p0.y + 0.022 },
      ];
      return {
        points: dotPts,
        fieldDirs: dotPts.map(() => ({ x: 0.64, y: 0.77 })),
        isClosed: false,
      };
    }
    if (!isClosed) {
      const lastPt = {
        x: ((lastIdx % W) - cx) / spanPx,
        y: (Math.floor(lastIdx / W) - cy) / spanPx,
      };
      if (
        pts.length === 0 ||
        Math.hypot(lastPt.x - pts[pts.length - 1].x, lastPt.y - pts[pts.length - 1].y) > 0.004
      ) {
        pts.push(lastPt);
      }
    }
    const smoothed = chaikinSmooth(pts, 2, isClosed);
    if (isClosed && smoothed.length >= 3) {
      smoothed.push({ ...smoothed[0] });
      smoothed.push({ ...smoothed[1] });
    }

    // Orient open strokes cleanly along the integrated Calligraphic Vector Field flow
    if (!isClosed && smoothed.length >= 2) {
      let flowAlign = 0;
      for (let i = 1; i < smoothed.length; i++) {
        const dx = smoothed[i].x - smoothed[i - 1].x;
        const dy = smoothed[i].y - smoothed[i - 1].y;
        const fv = sampleVectorField(
          (smoothed[i].x + smoothed[i - 1].x) * 0.5,
          (smoothed[i].y + smoothed[i - 1].y) * 0.5,
        );
        flowAlign += dx * fv.x + dy * fv.y;
      }
      if (flowAlign < -0.02) {
        smoothed.reverse();
      }
    }

    const fieldDirs = smoothed.map((pt, i) => {
      const fv = sampleVectorField(pt.x, pt.y);
      const prevPt = smoothed[Math.max(0, i - 1)];
      const nextPt = smoothed[Math.min(smoothed.length - 1, i + 1)];
      let tx = nextPt.x - prevPt.x;
      let ty = nextPt.y - prevPt.y;
      const tl = Math.hypot(tx, ty) || 1;
      tx /= tl;
      ty /= tl;
      const mx = tx * 0.70 + fv.x * 0.30;
      const my = ty * 0.70 + fv.y * 0.30;
      const ml = Math.hypot(mx, my) || 1;
      return { x: mx / ml, y: my / ml };
    });

    return {
      points: smoothed,
      fieldDirs,
      isClosed,
    };
  });

  if (preferLeftToRight) {
    // English cursive order: sweep word left-to-right; defer tiny floating 'i' dots & high 't' crossbars to end
    strokes.sort((a, b) => {
      const ap = a.points;
      const bp = b.points;
      const lenA = Math.hypot(ap[ap.length - 1].x - ap[0].x, ap[ap.length - 1].y - ap[0].y);
      const lenB = Math.hypot(bp[bp.length - 1].x - bp[0].x, bp[bp.length - 1].y - bp[0].y);
      const isDotA = ap.length <= 3 && lenA < 0.07 ? 1 : 0;
      const isDotB = bp.length <= 3 && lenB < 0.07 ? 1 : 0;
      if (isDotA !== isDotB) return isDotA - isDotB;
      return ap[0].x - bp[0].x;
    });
  } else {
    // Chinese/Hangul/Kana stroke order: vector-field assisted top-to-leftmost sort with minimal-lift neighbor chaining
    strokes.sort((a, b) => {
      const a0 = a.points[0];
      const b0 = b.points[0];
      return a0.y * 1.18 + a0.x * 0.82 - (b0.y * 1.18 + b0.x * 0.82);
    });
  }

  let inkMinX = 0.45;
  let inkMaxX = -0.45;
  for (const s of strokes) {
    for (const p of s.points) {
      if (p.x < inkMinX) inkMinX = p.x;
      if (p.x > inkMaxX) inkMaxX = p.x;
    }
  }
  if (inkMinX > inkMaxX) {
    inkMinX = -0.35;
    inkMaxX = 0.35;
  }

  const firstStroke = strokes[0];
  const lastStroke = strokes[strokes.length - 1];
  const entryPt = firstStroke ? firstStroke.points[0] : { x: -0.25, y: 0 };
  const exitPt = lastStroke
    ? lastStroke.points[lastStroke.points.length - 1]
    : { x: 0.25, y: 0 };

  return {
    strokes,
    inkMinX,
    inkMaxX,
    entryPt,
    exitPt,
    strokeCount: strokes.length,
  };
}

function extractOrderedGlyphStrokes(
  char: string,
  font: string,
  skelCanvas: HTMLCanvasElement,
  skelCtx: CanvasRenderingContext2D,
): ExtractedGlyph {
  const key = `G|${char}|${font.split('px ')[1] ?? font}`;
  const cached = skeletonCache.get(key);
  if (cached) return cached;

  const S = SKEL_GRID;
  skelCanvas.width = S;
  skelCanvas.height = S;
  skelCtx.clearRect(0, 0, S, S);
  const drawSize = Math.round(S * 0.72);
  const drawFont = font.replace(/\d+px/, `${drawSize}px`);
  skelCtx.font = drawFont;
  skelCtx.textAlign = 'center';
  skelCtx.textBaseline = 'middle';
  skelCtx.fillStyle = '#ffffff';
  // Pure fillText only (no strokeText bloat) preserves crisp open interior loops & CJK stroke separation
  skelCtx.fillText(char, S / 2, S / 2);

  const img = skelCtx.getImageData(0, 0, S, S).data;
  const grid = new Uint8Array(S * S);
  let litCount = 0;
  for (let i = 0; i < S * S; i++) {
    if (img[i * 4 + 3] > 76) {
      grid[i] = 1;
      litCount++;
    }
  }
  if (litCount === 0) {
    return {
      strokes: [],
      inkMinX: -0.3,
      inkMaxX: 0.3,
      entryPt: { x: -0.2, y: 0 },
      exitPt: { x: 0.2, y: 0 },
      strokeCount: 0,
    };
  }

  const glyph = thinAndExtractStrokes(grid, S, S, Math.max(28, drawSize), S / 2, S / 2, false);
  if (glyph.strokes.length > 0) {
    if (skeletonCache.size > 480) skeletonCache.clear();
    skeletonCache.set(key, glyph);
  }
  return glyph;
}

// Whole-word flowing cursive English extractor ("Caveat" connected word strips)
function extractCursiveWordStrokes(
  word: string,
  font: string,
  skelCanvas: HTMLCanvasElement,
  skelCtx: CanvasRenderingContext2D,
): ExtractedGlyph {
  const key = `W|${word}|${font.split('px ')[1] ?? font}`;
  const cached = skeletonCache.get(key);
  if (cached) return cached;

  const H = SKEL_GRID;
  const drawSize = Math.round(H * 0.66);
  const drawFont = font.replace(/\d+px/, `${drawSize}px`);
  skelCtx.font = drawFont;
  const measuredW = skelCtx.measureText(word).width;
  const W = Math.max(H, Math.min(720, Math.ceil(measuredW + H * 0.55)));

  skelCanvas.width = W;
  skelCanvas.height = H;
  skelCtx.clearRect(0, 0, W, H);
  skelCtx.font = drawFont;
  skelCtx.textAlign = 'center';
  skelCtx.textBaseline = 'middle';
  skelCtx.fillStyle = '#ffffff';
  // Pure fillText preserves crisp lowercase counters ('o','e','a','g','d') without plugging loops
  skelCtx.fillText(word, W / 2, H / 2);

  const img = skelCtx.getImageData(0, 0, W, H).data;
  const grid = new Uint8Array(W * H);
  let litCount = 0;
  for (let i = 0; i < W * H; i++) {
    if (img[i * 4 + 3] > 72) {
      grid[i] = 1;
      litCount++;
    }
  }
  if (litCount === 0) {
    return {
      strokes: [],
      inkMinX: -0.4,
      inkMaxX: 0.4,
      entryPt: { x: -0.3, y: 0 },
      exitPt: { x: 0.3, y: 0 },
      strokeCount: 0,
    };
  }

  const glyph = thinAndExtractStrokes(grid, W, H, Math.max(28, drawSize), W / 2, H / 2, true);
  if (glyph.strokes.length > 0) {
    if (skeletonCache.size > 480) skeletonCache.clear();
    skeletonCache.set(key, glyph);
  }
  return glyph;
}

export function createFogSimulation(): FogSimulationSystem {
  preloadHandwritingFonts();
  let currentMirrorAspect = 1.5;
  let wipeDims = computeAspectDims(currentMirrorAspect, WIPE_MAX_DIM);
  let dropDims = computeAspectDims(currentMirrorAspect, DROPLET_MAX_DIM);
  let wetDims = computeAspectDims(currentMirrorAspect, WET_FILM_MAX_DIM);

  const wipeCanvas = document.createElement('canvas');
  wipeCanvas.width = wipeDims.w;
  wipeCanvas.height = wipeDims.h;
  const wipeCtx = wipeCanvas.getContext('2d')!;
  wipeCtx.fillStyle = '#000000';
  wipeCtx.fillRect(0, 0, wipeDims.w, wipeDims.h);

  const wipeTexture = new THREE.CanvasTexture(wipeCanvas);
  wipeTexture.minFilter = THREE.LinearFilter;
  wipeTexture.magFilter = THREE.LinearFilter;

  const dropletCanvas = document.createElement('canvas');
  dropletCanvas.width = dropDims.w;
  dropletCanvas.height = dropDims.h;
  const dropletCtx = dropletCanvas.getContext('2d')!;
  dropletCtx.fillStyle = '#000000';
  dropletCtx.fillRect(0, 0, dropDims.w, dropDims.h);

  const dropletTexture = new THREE.CanvasTexture(dropletCanvas);
  dropletTexture.minFilter = THREE.LinearFilter;
  dropletTexture.magFilter = THREE.LinearFilter;

  // Pre-rendered droplet dome sprites (gravity-sagged sessile dome). Drawing a cached sprite
  // is far cheaper than building a 6-stop radial gradient per bead per frame.
  const makeBeadSprite = (lobeShiftFrac: number): HTMLCanvasElement => {
    const S = 64;
    const R = S / 2;
    const c = document.createElement('canvas');
    c.width = S;
    c.height = S;
    const g = c.getContext('2d')!;
    const grad = g.createRadialGradient(R, R + lobeShiftFrac * 0.72 * R, R * 0.08, R, R, R);
    grad.addColorStop(0.0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.35, 'rgba(255, 255, 255, 0.94)');
    grad.addColorStop(0.62, 'rgba(255, 255, 255, 0.78)');
    grad.addColorStop(0.84, 'rgba(255, 255, 255, 0.46)');
    grad.addColorStop(0.96, 'rgba(255, 255, 255, 0.12)');
    grad.addColorStop(1.0, 'rgba(255, 255, 255, 0)');
    g.fillStyle = grad;
    g.beginPath();
    g.arc(R, R, R, 0, Math.PI * 2);
    g.fill();
    return c;
  };
  const beadSpriteStatic = makeBeadSprite(0.06);
  const beadSpriteRolling = makeBeadSprite(0.22);

  const wetFilmCanvas = document.createElement('canvas');
  wetFilmCanvas.width = wetDims.w;
  wetFilmCanvas.height = wetDims.h;
  const wetFilmCtx = wetFilmCanvas.getContext('2d')!;
  wetFilmCtx.fillStyle = '#000000';
  wetFilmCtx.fillRect(0, 0, wetDims.w, wetDims.h);

  const wetFilmTexture = new THREE.CanvasTexture(wetFilmCanvas);
  wetFilmTexture.minFilter = THREE.LinearFilter;
  wetFilmTexture.magFilter = THREE.LinearFilter;

  const screenMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uSharpTex: { value: null },
      uNearBlurTex: { value: null },
      uBlurTex: { value: null },
      uWipeMap: { value: wipeTexture },
      uDropletMap: { value: dropletTexture },
      uWetFilmMap: { value: wetFilmTexture },
      uFluidLayerTex: { value: null },
      uStrokeDepthTex: { value: null },
      uWipeTexSize: { value: new THREE.Vector2(wipeDims.w, wipeDims.h) },
      uDropTexSize: { value: new THREE.Vector2(dropDims.w, dropDims.h) },
      uWetTexSize: { value: new THREE.Vector2(wetDims.w, wetDims.h) },
      uTime: { value: 0 },
      uFogLevel: { value: 0.92 },
      uCoverage: { value: 1.0 },
      uTempDiff: { value: 0.65 },
      uMirrorAspect: { value: currentMirrorAspect },
      uAudioEnergy: { value: 0 },
      uBeatPulse: { value: 0 },
      uParallax: { value: new THREE.Vector2(0, 0) },
    },
    vertexShader: QUAD_VERTEX_SHADER,
    fragmentShader: FROSTED_MIRROR_FRAGMENT_SHADER,
    depthTest: false,
    depthWrite: false,
  });

  const particleCount = 34;
  const positions = new Float32Array(particleCount * 3);
  const velocities = new Float32Array(particleCount * 2);
  const phases = new Float32Array(particleCount);
  for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 2.2;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 2.2;
    positions[i * 3 + 2] = 0;
    velocities[i * 2] = (Math.random() - 0.5) * 0.03;
    velocities[i * 2 + 1] = 0.14 + Math.random() * 0.18;
    phases[i] = Math.random() * Math.PI * 2;
  }

  const steamGeo = new THREE.BufferGeometry();
  steamGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const steamTex = createSteamParticleTexture();
  const steamMat = new THREE.PointsMaterial({
    map: steamTex,
    size: 0.34,
    transparent: true,
    opacity: 0.12,
    depthWrite: false,
    depthTest: false,
    blending: THREE.NormalBlending,
  });
  const steamParticles = new THREE.Points(steamGeo, steamMat);

  // 96x96 CPU Edge-Tension & Tangential Flow Field
  const gridCount = WET_GRID_SIZE * WET_GRID_SIZE;
  const wetAmount = new Float32Array(gridCount);
  const wetDirX = new Float32Array(gridCount);
  const wetDirY = new Float32Array(gridCount);
  const wetForce = new Float32Array(gridCount);

  let currentHandwritingScale = 1.05;
  let currentStrokeWeight = 0.36; // Default: 細指
  let breathTimer = 0;
  let manualClearHold = false;
  let currentFogLevel = 0.72;
  let currentTempDiff = 0.65;
  let burstBoost = 0;
  let beatPulse = 0;
  let wipeRecoverTimer = 0;
  let approximateWipedRatio = 0;
  let wipeDirty = false;
  let wetFilmDirty = false;
  let dropletHasContent = true; // forces one clear + upload on first frame
  let dropletDirty = false;
  let fluidHold = 3;
  let wipeFadeDebt = 0;
  let wetFadeDebt = 0;
  const beadOrder: number[] = [];
  let writingZoneIndex = 0;
  let writeSequence = 0;
  let activeJob: ActiveHandwritingJob | null = null;
  const activeBeads: EdgeWaterBead[] = [];

  const measureCanvas = document.createElement('canvas');
  const measureCtx = measureCanvas.getContext('2d')!;
  const skelCanvas = document.createElement('canvas');
  skelCanvas.width = SKEL_GRID;
  skelCanvas.height = SKEL_GRID;
  const skelCtx = skelCanvas.getContext('2d', { willReadFrequently: true })!;

  const stampWetField = (
    u: number,
    v: number,
    radiusV: number,
    strength = 1.0,
    dirX = 0,
    dirY = -1,
    pressure = 0.5,
  ) => {
    const gxC = Math.floor(THREE.MathUtils.clamp(u, 0, 0.999) * WET_GRID_SIZE);
    const gyC = Math.floor(THREE.MathUtils.clamp(v, 0, 0.999) * WET_GRID_SIZE);
    const radCells = Math.max(1, Math.ceil(radiusV * WET_GRID_SIZE * 1.45));
    const dirLen = Math.hypot(dirX, dirY);
    const ux = dirLen > 0.001 ? dirX / dirLen : 0;
    const uy = dirLen > 0.001 ? dirY / dirLen : -1;

    for (let dy = -radCells; dy <= radCells; dy++) {
      const ny = gyC + dy;
      if (ny < 0 || ny >= WET_GRID_SIZE) continue;
      for (let dx = -radCells; dx <= radCells; dx++) {
        const nx = gxC + dx;
        if (nx < 0 || nx >= WET_GRID_SIZE) continue;
        const d = Math.hypot(dx, dy) / radCells;
        if (d <= 1) {
          const w = (1 - d) * strength;
          const idx = ny * WET_GRID_SIZE + nx;
          wetAmount[idx] = Math.min(1, Math.max(wetAmount[idx], w));
          wetForce[idx] = Math.min(1, Math.max(wetForce[idx], pressure * (1 - d * 0.5)));
          wetDirX[idx] = wetDirX[idx] * 0.45 + ux * 0.55;
          wetDirY[idx] = wetDirY[idx] * 0.45 + uy * 0.55;
        }
      }
    }
  };

  const sampleFieldBilinear = (
    arr: Float32Array,
    u: number,
    v: number,
  ): number => {
    const fx = THREE.MathUtils.clamp(u, 0, 0.999) * (WET_GRID_SIZE - 1);
    const fy = THREE.MathUtils.clamp(v, 0, 0.999) * (WET_GRID_SIZE - 1);
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(WET_GRID_SIZE - 1, x0 + 1);
    const y1 = Math.min(WET_GRID_SIZE - 1, y0 + 1);
    const tx = fx - x0;
    const ty = fy - y0;
    const v00 = arr[y0 * WET_GRID_SIZE + x0];
    const v10 = arr[y0 * WET_GRID_SIZE + x1];
    const v01 = arr[y1 * WET_GRID_SIZE + x0];
    const v11 = arr[y1 * WET_GRID_SIZE + x1];
    return THREE.MathUtils.lerp(
      THREE.MathUtils.lerp(v00, v10, tx),
      THREE.MathUtils.lerp(v01, v11, tx),
      ty,
    );
  };

  const spawnEdgeBead = (
    u: number,
    v: number,
    vx = 0,
    vy = 0,
    r = 0.0022,
    rolling = false,
  ) => {
    if (u < 0.01 || u > 0.99 || v < 0.01 || v > 0.99) return;
    if (activeBeads.length >= 240) {
      let oldestStaticIdx = -1;
      for (let i = 0; i < activeBeads.length; i++) {
        if (!activeBeads[i].rolling) {
          oldestStaticIdx = i;
          break;
        }
      }
      if (oldestStaticIdx >= 0) {
        activeBeads.splice(oldestStaticIdx, 1);
      } else {
        return;
      }
    }
    activeBeads.push({
      u,
      v,
      prevU: u,
      prevV: v,
      vx,
      vy,
      r: THREE.MathUtils.clamp(r, 0.001, 0.0085),
      life: 0,
      maxLife: rolling ? 2.0 + Math.random() * 1.8 : 3.6 + Math.random() * 2.0,
      seed: Math.random() * 14,
      slipPhase: Math.random() * Math.PI * 2,
      rolling,
    });
  };

  const paintDropletOnCleanCanvas = (
    u: number,
    v: number,
    radiusV: number,
    intensity: number,
    elongation = 1.18,
  ) => {
    const cx = u * dropDims.w;
    const cy = (1 - v) * dropDims.h;
    const radPx = Math.max(1.8, radiusV * dropDims.h);

    dropletCtx.save();
    dropletCtx.translate(cx, cy);
    dropletCtx.scale(1, elongation);
    dropletCtx.globalAlpha = THREE.MathUtils.clamp(intensity, 0, 1);
    dropletCtx.imageSmoothingQuality = 'medium';
    dropletCtx.drawImage(
      elongation > 1.12 ? beadSpriteRolling : beadSpriteStatic,
      -radPx,
      -radPx,
      radPx * 2,
      radPx * 2,
    );
    dropletCtx.restore();
  };

  const paintWetFilmStamp = (
    u: number,
    v: number,
    radiusV: number,
    wetStrength: number,
    mistPileShoulder: number,
  ) => {
    const cx = u * wetDims.w;
    const cy = (1 - v) * wetDims.h;
    const radPx = Math.max(2.0, radiusV * wetDims.h * 1.45);

    const grad = wetFilmCtx.createRadialGradient(
      cx,
      cy,
      radPx * 0.12,
      cx,
      cy,
      radPx,
    );
    // R = Surface Wetness Coefficient; G = displaced mist pile-up ridge on outer shoulder!
    grad.addColorStop(0, `rgba(255, 0, 0, ${wetStrength})`);
    grad.addColorStop(0.55, `rgba(255, 45, 0, ${wetStrength * 0.72})`);
    grad.addColorStop(0.76, `rgba(220, 255, 0, ${mistPileShoulder})`);
    grad.addColorStop(1, 'rgba(200, 255, 0, 0)');
    wetFilmCtx.fillStyle = grad;
    wetFilmCtx.beginPath();
    wetFilmCtx.arc(cx, cy, radPx, 0, Math.PI * 2);
    wetFilmCtx.fill();
    wetFilmDirty = true;
  };

  const paintWipeCircle = (
    u: number,
    v: number,
    radiusV: number,
    opacity = 0.98,
    dirX = 0,
    dirY = -1,
    pressure = 0.5,
  ) => {
    const cx = u * wipeDims.w;
    const cy = (1 - v) * wipeDims.h;
    const radiusPx = Math.max(2.2, radiusV * wipeDims.h);

    wipeCtx.save();
    const grad = wipeCtx.createRadialGradient(
      cx,
      cy,
      radiusPx * 0.30,
      cx,
      cy,
      radiusPx,
    );
    grad.addColorStop(0, `rgba(255, 255, 255, ${opacity})`);
    grad.addColorStop(0.70, `rgba(255, 255, 255, ${opacity * 0.55})`);
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    wipeCtx.fillStyle = grad;
    wipeCtx.beginPath();
    wipeCtx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
    wipeCtx.fill();
    wipeCtx.restore();

    paintWetFilmStamp(
      u,
      v,
      radiusV,
      0.34 * opacity,
      Math.min(0.88, (0.35 + currentFogLevel * 0.52) * pressure),
    );
    stampWetField(u, v, radiusV, opacity, dirX, dirY, pressure);
    wipeDirty = true;
    approximateWipedRatio = Math.min(1, approximateWipedRatio + 0.005);
  };

  // Surface Tension parameter (controls capillary pinning vs gravitational detachment)
  const SURFACE_TENSION = 0.44;

  // Unified Fluid Displacement & Particle Accumulation Physics Engine
  const fluidDisplacementStroke = (
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    brushRadius: number,
    speedNorm: number,
    isStrokeEnd = false,
    fingerPressure = 0.55,
    isHandwriting = false,
  ) => {
    const dxIso = (u1 - u0) * currentMirrorAspect;
    const dyIso = v1 - v0;
    const dist = Math.hypot(dxIso, dyIso);
    if (dist < 0.00015) return;

    const invD = 1 / dist;
    const tx = dxIso * invD;
    const ty = dyIso * invD;
    const nx = -ty;
    const ny = tx;
    const downwardSlope = THREE.MathUtils.clamp(-ty, 0, 1);

    const steps = Math.max(
      1,
      Math.ceil((dist * wipeDims.h) / Math.max(1.8, brushRadius * wipeDims.h * 0.36)),
    );

    const slowFactor = THREE.MathUtils.clamp(1.38 - speedNorm * 0.58, 0.35, 1.38);
    const strokeForce = THREE.MathUtils.clamp(
      0.28 + fingerPressure * 0.45 + slowFactor * 0.22 + downwardSlope * 0.16,
      0.30,
      1.0,
    );
    const baseSpawnProb = isHandwriting
      ? THREE.MathUtils.clamp(0.025 + downwardSlope * 0.045 + (isStrokeEnd ? 0.16 : 0), 0.02, 0.20)
      : THREE.MathUtils.clamp(
          0.09 + speedNorm * 0.07 + fingerPressure * 0.10 + downwardSlope * 0.14 + (isStrokeEnd ? 0.32 : 0),
          0.08,
          0.52,
        );

    let crossedExistingPath = false;

    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const u = THREE.MathUtils.lerp(u0, u1, t);
      const v = THREE.MathUtils.lerp(v0, v1, t);

      if (!isHandwriting) {
        const priorWet = sampleFieldBilinear(wetAmount, u, v);
        if (!crossedExistingPath && priorWet > 0.34 && priorWet < 0.92 && i > 0) {
          crossedExistingPath = true;
          if (currentFogLevel > 0.22) {
            const convRadius = THREE.MathUtils.clamp(
              (0.0031 + currentFogLevel * 0.0014) * (0.72 + currentStrokeWeight * 0.40),
              0.0024,
              0.0058,
            );
            spawnEdgeBead(u, v, tx * 0.012, -0.014, convRadius, true);
          }
        }
      }

      const effectiveR = brushRadius * (isHandwriting ? 0.96 + fingerPressure * 0.18 : 0.88 + fingerPressure * 0.26);
      paintWipeCircle(u, v, effectiveR, isHandwriting ? 0.99 : 0.96, tx, ty, strokeForce);

      const isLast = isStrokeEnd && i === steps;
      const stride = isHandwriting ? 7 : 4;
      if (
        currentFogLevel > 0.24 &&
        ((i % stride === 0 && Math.random() < baseSpawnProb) || (isLast && Math.random() < (isHandwriting ? 0.24 : 0.82)))
      ) {
        const preferLower = ny <= 0 ? 1 : -1;
        const side = Math.random() < 0.76 ? preferLower : -preferLower;
        const rimOffset = effectiveR * (0.98 + Math.random() * 0.22);
        const jitter = (Math.random() - 0.5) * effectiveR * 0.18;
        const bu = isLast
          ? u
          : u + (nx * side * (rimOffset + jitter)) / Math.max(0.35, currentMirrorAspect);
        const bv = isLast ? v - effectiveR * 0.85 : v + ny * side * (rimOffset + jitter);

        if (isHandwriting) {
          // Ultra-restrained stationary micro-beads only so drips never slash across letters mid-word
          if (!isLast && i % 11 === 0 && Math.random() < 0.045) {
            const hRadius = (0.0008 + Math.random() * 0.0004) * (0.85 + currentStrokeWeight * 0.2);
            spawnEdgeBead(bu, bv, 0, -0.001, hRadius, false);
          }
        } else {
          const weightScale = 0.62 + currentStrokeWeight * 0.44;
          const speedSizeMult = THREE.MathUtils.clamp(1.24 - speedNorm * 0.40 + fingerPressure * 0.16, 0.58, 1.48);
          const slopePoolBonus = 1 + downwardSlope * 0.35;
          const isMicro = !isLast && Math.random() < 0.52;
          const beadRadius =
            (isMicro
              ? 0.0012 + Math.random() * 0.0010
              : 0.0023 + currentFogLevel * 0.0014 + Math.random() * 0.0011) *
            weightScale *
            speedSizeMult *
            (isLast ? slopePoolBonus : 1);

          const releaseCriterion = SURFACE_TENSION * 0.0068 * (1.0 - downwardSlope * 0.38);
          const startsRolling =
            (isLast && (downwardSlope > 0.25 || beadRadius >= releaseCriterion) && Math.random() < 0.78) ||
            (!isMicro && beadRadius >= releaseCriterion * 1.05 && Math.random() < 0.45);

          const kickVx = nx * side * 0.005 + tx * speedNorm * 0.014;
          const kickVy = ty * speedNorm * 0.012 - (startsRolling ? 0.014 : 0.005);

          spawnEdgeBead(bu, bv, kickVx, kickVy, beadRadius, startsRolling);
        }
      }
    }
  };

  const buildGlyphFromTrueOrder = (trueG: TrueStrokeGlyph): ExtractedGlyph => {
    const strokes: RawStrokeData[] = [];
    let inkMinX = 0.45;
    let inkMaxX = -0.45;

    for (const rawPts of trueG.strokes) {
      if (rawPts.length < 2) continue;
      const pts: Pt2[] = rawPts.map(([x, y]) => ({ x, y }));
      const smoothed = pts.length >= 3 ? chaikinSmooth(pts, 1, false) : pts;
      const fieldDirs: Pt2[] = smoothed.map((_, idx) => {
        const p0 = smoothed[Math.max(0, idx - 1)];
        const p1 = smoothed[Math.min(smoothed.length - 1, idx + 1)];
        const dx = p1.x - p0.x;
        const dy = p1.y - p0.y;
        const l = Math.hypot(dx, dy) || 1;
        return { x: dx / l, y: dy / l };
      });
      for (const pt of smoothed) {
        if (pt.x < inkMinX) inkMinX = pt.x;
        if (pt.x > inkMaxX) inkMaxX = pt.x;
      }
      strokes.push({ points: smoothed, fieldDirs, isClosed: false });
    }

    if (inkMinX > inkMaxX) {
      inkMinX = -trueG.advance * 0.4;
      inkMaxX = trueG.advance * 0.4;
    }

    const first = strokes[0];
    const last = strokes[strokes.length - 1];
    return {
      strokes,
      inkMinX,
      inkMaxX,
      entryPt: first ? first.points[0] : { x: -0.2, y: 0 },
      exitPt: last ? last.points[last.points.length - 1] : { x: 0.2, y: 0 },
      strokeCount: strokes.length,
    };
  };

  const prepareHandwritingJob = (
    text: string,
    durationSec = 0.9,
    isChorus = false,
  ) => {
    const clean = text.trim();
    if (!clean) return;

    manualClearHold = false;
    burstBoost = Math.max(burstBoost, isChorus ? 0.92 : 0.75);

    const canvasW = wipeDims.w;
    const canvasH = wipeDims.h;
    const shortSide = Math.min(canvasW, canvasH);
    const safeBoxW = canvasW * 0.86;

    const charCount = Array.from(clean).length;
    let baseRatio = 0.11;
    if (charCount <= 4) {
      baseRatio = 0.185;
    } else if (charCount <= 8) {
      baseRatio = 0.148;
    } else if (charCount <= 14) {
      baseRatio = 0.122;
    } else {
      baseRatio = 0.105;
    }

    const chorusMult = isChorus ? 1.08 : 1.0;
    let fontSize = Math.round(
      THREE.MathUtils.clamp(
        shortSide * baseRatio * currentHandwritingScale * chorusMult,
        52,
        240,
      ),
    );

    let cjkTrackingPx = fontSize * 0.12;
    let latinGapPx = fontSize * 0.18;
    let lines = splitBalancedLines(
      measureCtx,
      clean,
      safeBoxW,
      fontSize,
      cjkTrackingPx,
      latinGapPx,
    );

    for (let attempt = 0; attempt < 6; attempt++) {
      const maxLineW = Math.max(
        ...lines.map((ln) =>
          measureMixedLineWidth(
            measureCtx,
            tokenizeMixedLine(ln),
            fontSize,
            cjkTrackingPx,
            latinGapPx,
          ),
        ),
      );
      if (maxLineW <= safeBoxW || fontSize <= 48) break;
      const shrink = Math.max(0.76, (safeBoxW / Math.max(1, maxLineW)) * 0.96);
      fontSize = Math.max(48, Math.round(fontSize * shrink));
      cjkTrackingPx = fontSize * 0.12;
      latinGapPx = fontSize * 0.18;
      lines = splitBalancedLines(
        measureCtx,
        clean,
        safeBoxW,
        fontSize,
        cjkTrackingPx,
        latinGapPx,
      );
    }

    const lineHeight = fontSize * 1.42;
    const totalH = lines.length * lineHeight;
    const startY = -totalH / 2 + lineHeight / 2;

    const zoneYFracs =
      currentMirrorAspect < 0.85 ? [0.36, 0.64] : [0.38, 0.62];
    const zoneY = zoneYFracs[writingZoneIndex % zoneYFracs.length];
    writingZoneIndex += 1;

    const jitterX = (Math.random() - 0.5) * (canvasW * 0.02);
    const targetX = canvasW / 2 + jitterX;
    const targetY = canvasH * zoneY;
    // Cohesive natural mirror-writing wrist slant across the phrase
    const lineTiltRad = (Math.random() - 0.5) * 0.052;
    const lineCos = Math.cos(lineTiltRad);
    const lineSin = Math.sin(lineTiltRad);

    const fontNormH = fontSize / canvasH;
    const weightT = THREE.MathUtils.clamp((currentStrokeWeight - 0.15) / (1.6 - 0.15), 0, 1);
    // Crisp proportional finger radius so interior counters & dense strokes stay clear and distinct
    const handBrushRadius = THREE.MathUtils.clamp(
      fontNormH * THREE.MathUtils.lerp(0.046, 0.086, weightT),
      0.0036,
      Math.max(0.0070, fontNormH * 0.095),
    );

    const builtStrokes: HandwritingStroke[] = [];

    interface PreparedChar {
      ch: string;
      isSpace: boolean;
      wordIdx: number;
      script: ScriptKind;
      glyph: ExtractedGlyph | null;
      glyphBoxPx: number;
      advancePx: number;
      gapAfterPx: number;
      slantSkew: number;
    }

    lines.forEach((line, lineIdx) => {
      const tokens = tokenizeMixedLine(line);
      const prepared: PreparedChar[] = [];

      tokens.forEach((tok, tokIdx) => {
        if (!tok.trim()) {
          prepared.push({
            ch: ' ',
            isSpace: true,
            wordIdx: tokIdx,
            script: 'latin',
            glyph: null,
            glyphBoxPx: fontSize,
            advancePx: fontSize * 0.36,
            gapAfterPx: 0,
            slantSkew: 0,
          });
          return;
        }

        const chars = Array.from(tok);
        for (const ch of chars) {
          const trueG = getTrueStrokeOrderGlyph(ch);
          const script: ScriptKind = trueG ? trueG.script : detectScriptKind(ch);
          const tokFont = resolveHandwritingFont(ch, fontSize);
          measureCtx.font = tokFont;
          const slantSkew =
            script === 'latin' ? -0.14 : -0.07 + (Math.random() - 0.5) * 0.03;
          const charScale =
            script === 'latin'
              ? 0.98 + (Math.random() - 0.5) * 0.04
              : 0.95 + (Math.random() - 0.5) * 0.08;
          const glyphBoxPx = fontSize * charScale;

          const glyph = trueG
            ? buildGlyphFromTrueOrder(trueG)
            : extractOrderedGlyphStrokes(ch, tokFont, skelCanvas, skelCtx);

          const advancePx = trueG
            ? trueG.advance * fontSize
            : Math.max(measureCtx.measureText(ch).width, fontSize * 0.55);

          prepared.push({
            ch,
            isSpace: false,
            wordIdx: tokIdx,
            script,
            glyph,
            glyphBoxPx,
            advancePx,
            gapAfterPx: 0,
            slantSkew,
          });
        }
      });

      // Dynamic script & stroke-order optical kerning (zero overlapping, balanced spacing)
      let totalWidth = 0;
      for (let i = 0; i < prepared.length; i++) {
        const cur = prepared[i];
        if (!cur.isSpace && i < prepared.length - 1) {
          const nxt = prepared[i + 1];
          if (!nxt.isSpace) {
            const isPunct = /^[，。、！？,.!?;:·'’"]$/.test(nxt.ch);
            if (isPunct) {
              cur.gapAfterPx = fontSize * 0.02;
            } else if (cur.script === 'latin' && nxt.script === 'latin') {
              // Within same English word: tighten cleanly without colliding letter bodies
              cur.gapAfterPx = cur.wordIdx === nxt.wordIdx ? fontSize * 0.035 : fontSize * 0.14;
            } else if (cur.script !== nxt.script) {
              // Mixed CJK <-> Latin optical clearance
              cur.gapAfterPx = fontSize * 0.16;
            } else {
              // CJK / Hangul / Kana: uniform breathing room tuned subtly by stroke density
              const avgStrokes =
                ((cur.glyph?.strokeCount ?? 5) + (nxt.glyph?.strokeCount ?? 5)) * 0.5;
              const densityMult = THREE.MathUtils.clamp(0.84 + avgStrokes * 0.025, 0.86, 1.14);
              cur.gapAfterPx = fontSize * 0.095 * densityMult;
            }
          }
        }
        totalWidth += cur.advancePx + cur.gapAfterPx;
      }

      let cursorX = -totalWidth / 2;
      const linePhase = Math.random() * Math.PI * 2;
      const cy = startY + lineIdx * lineHeight;

      for (let i = 0; i < prepared.length; i++) {
        const item = prepared[i];
        if (item.isSpace || !item.glyph) {
          cursorX += item.advancePx;
          continue;
        }

        const { script, glyph, glyphBoxPx, slantSkew } = item;
        const gx = cursorX + item.advancePx / 2;
        // Gentle coherent baseline drift (smaller for Latin so word baselines stay crisp & aligned)
        const waveY =
          Math.sin(i * 0.45 + linePhase) *
          (fontSize * (script === 'latin' ? 0.018 : 0.038));
        const jitterY =
          (Math.random() - 0.5) * (fontSize * (script === 'latin' ? 0.012 : 0.030));
        const gy = cy + waveY + jitterY;
        const charRot =
          script === 'latin'
            ? (Math.random() - 0.5) * 0.035
            : (Math.random() - 0.5) * 0.068;
        const charCos = Math.cos(charRot);
        const charSin = Math.sin(charRot);

        cursorX += item.advancePx + item.gapAfterPx;

        const rawStrokes = glyph.strokes;
        for (let sIdx = 0; sIdx < rawStrokes.length; sIdx++) {
          const raw = rawStrokes[sIdx];
          if (raw.points.length < 2) continue;

          const bowPhase = Math.random() * Math.PI * 2;
          const bowAmp = (Math.random() - 0.5) * (script === 'latin' ? 0.014 : 0.026);
          const uvPts: Array<{ u: number; v: number }> = [];
          const uvFieldDirs: Array<{ u: number; v: number }> = [];
          const cumLengths: number[] = [0];
          let totalLength = 0;

          for (let p = 0; p < raw.points.length; p++) {
            const tNorm = p / Math.max(1, raw.points.length - 1);
            const pt = raw.points[p];
            const fDir = raw.fieldDirs[p] ?? { x: 1, y: 0 };

            const nx = pt.x + slantSkew * pt.y;
            const ny = pt.y + Math.sin(tNorm * Math.PI + bowPhase) * bowAmp;

            const pxLocal = nx * glyphBoxPx;
            const pyLocal = ny * glyphBoxPx;

            const cxLocal = gx + (pxLocal * charCos - pyLocal * charSin);
            const cyLocal = gy + (pxLocal * charSin + pyLocal * charCos);

            const pxScreen = targetX + (cxLocal * lineCos - cyLocal * lineSin);
            const pyScreen = targetY + (cxLocal * lineSin + cyLocal * lineCos);

            const u = THREE.MathUtils.clamp(pxScreen / canvasW, 0.02, 0.98);
            const v = THREE.MathUtils.clamp(1 - pyScreen / canvasH, 0.02, 0.98);

            const sx = fDir.x + slantSkew * fDir.y;
            const sy = fDir.y;
            const crx = sx * charCos - sy * charSin;
            const cry = sx * charSin + sy * charCos;
            const lrx = crx * lineCos - cry * lineSin;
            const lry = crx * lineSin + cry * lineCos;
            const fl = Math.hypot(lrx, lry) || 1;
            uvFieldDirs.push({
              u: (lrx / fl) / Math.max(0.35, currentMirrorAspect),
              v: -(lry / fl),
            });

            if (uvPts.length > 0) {
              const prev = uvPts[uvPts.length - 1];
              const segLen = Math.hypot((u - prev.u) * currentMirrorAspect, v - prev.v);
              totalLength += segLen;
              cumLengths.push(totalLength);
            }
            uvPts.push({ u, v });
          }

          if (uvPts.length >= 2) {
            const curvatures: number[] = new Array(uvPts.length).fill(0);
            for (let k = 1; k < uvPts.length - 1; k++) {
              const a = uvPts[k - 1];
              const b = uvPts[k];
              const c = uvPts[k + 1];
              const v1x = (b.u - a.u) * currentMirrorAspect;
              const v1y = b.v - a.v;
              const v2x = (c.u - b.u) * currentMirrorAspect;
              const v2y = c.v - b.v;
              const l1 = Math.hypot(v1x, v1y) || 1e-5;
              const l2 = Math.hypot(v2x, v2y) || 1e-5;
              const dot = THREE.MathUtils.clamp((v1x * v2x + v1y * v2y) / (l1 * l2), -1, 1);
              curvatures[k] = (1 - dot) * 0.5;
            }
            const cleanLen = Math.max(0.002, totalLength);
            const isSmallDetail = uvPts.length <= 3 && cleanLen < 0.014;

            let linkToPrev: 'lift' | 'ligature' | 'flow' = 'lift';
            if (builtStrokes.length > 0 && !isSmallDetail) {
              const prevS = builtStrokes[builtStrokes.length - 1];
              const prevEnd = prevS.points[prevS.points.length - 1];
              const currStart = uvPts[0];
              const jumpDist = Math.hypot(
                (currStart.u - prevEnd.u) * currentMirrorAspect,
                currStart.v - prevEnd.v,
              );
              const prevItem = prepared[Math.max(0, i - 1)];
              const sameEnglishWord =
                script === 'latin' &&
                prevItem &&
                !prevItem.isSpace &&
                prevItem.wordIdx === item.wordIdx;
              if (sameEnglishWord && sIdx === 0 && jumpDist < handBrushRadius * 6.0) {
                linkToPrev = 'ligature';
              } else if (jumpDist < handBrushRadius * 8.0) {
                linkToPrev = 'flow';
              }
            }

            builtStrokes.push({
              points: uvPts,
              fieldDirs: uvFieldDirs,
              cumLengths,
              curvatures,
              totalLength: cleanLen,
              isSmallDetail,
              linkToPrev,
            });
          }
        }
      }
    });

    if (builtStrokes.length === 0) return;

    const sumLen = builtStrokes.reduce((acc, s) => acc + s.totalLength, 0);
    const targetTotalSec = THREE.MathUtils.clamp(
      Math.max(durationSec * 0.92, builtStrokes.length * 0.036),
      0.85,
      2.55,
    );
    const first = builtStrokes[0].points[0];

    activeJob = {
      strokes: builtStrokes,
      strokeIndex: 0,
      targetDist: 0,
      strokeElapsed: 0,
      tipU: first.u,
      tipV: first.v,
      tipVx: 0,
      tipVy: 0,
      brushRadius: handBrushRadius,
      strokeStarted: false,
      liftCooldownSec: 0,
      isLigatureJump: false,
      ligatureStartU: first.u,
      ligatureStartV: first.v,
      ligatureEndU: first.u,
      ligatureEndV: first.v,
      ligatureDurationSec: 0,
      ligatureTimer: 0,
      tipHoldSec: 0,
      baseSpeedUvPerSec: Math.max(0.95, sumLen / Math.max(0.45, targetTotalSec * 0.78)),
      pressure: 0.76,
      phaseSeed: Math.random() * 12,
      wordEndTriggered: false,
      finishHoldSec: 0,
    };
  };

  const sampleStrokeTarget = (
    stroke: HandwritingStroke,
    distAlong: number,
  ): { u: number; v: number; curvature: number; dirU: number; dirV: number } => {
    const clampedD = THREE.MathUtils.clamp(distAlong, 0, stroke.totalLength);
    const { cumLengths, points, curvatures, fieldDirs } = stroke;
    for (let i = 1; i < cumLengths.length; i++) {
      if (cumLengths[i] >= clampedD || i === cumLengths.length - 1) {
        const d0 = cumLengths[i - 1];
        const d1 = cumLengths[i];
        const seg = Math.max(1e-6, d1 - d0);
        const alpha = THREE.MathUtils.clamp((clampedD - d0) / seg, 0, 1);
        return {
          u: THREE.MathUtils.lerp(points[i - 1].u, points[i].u, alpha),
          v: THREE.MathUtils.lerp(points[i - 1].v, points[i].v, alpha),
          curvature: THREE.MathUtils.lerp(curvatures[i - 1], curvatures[i], alpha),
          dirU: THREE.MathUtils.lerp(fieldDirs[i - 1].u, fieldDirs[i].u, alpha),
          dirV: THREE.MathUtils.lerp(fieldDirs[i - 1].v, fieldDirs[i].v, alpha),
        };
      }
    }
    const last = points.length - 1;
    return {
      u: points[last].u,
      v: points[last].v,
      curvature: curvatures[last],
      dirU: fieldDirs[last].u,
      dirV: fieldDirs[last].v,
    };
  };

  // Real-Time Vector-Field & Ligature Spring-Damper Handwriting Solver
  const advanceHandwritingJob = (dt: number, elapsed: number) => {
    if (!activeJob) return;
    const job = activeJob;

    if (job.strokeIndex >= job.strokes.length) {
      if (!job.wordEndTriggered) {
        job.wordEndTriggered = true;
        // At most 2 slow gentle teardrops from downward stroke ends AFTER the phrase finishes
        let droppedCount = 0;
        for (let i = 0; i < job.strokes.length && droppedCount < 2; i++) {
          const s = job.strokes[i];
          if (Math.random() < 0.16 && currentFogLevel > 0.35) {
            const endPt = s.points[s.points.length - 1];
            const pPrev = s.points[Math.max(0, s.points.length - 2)];
            const slopeDown = THREE.MathUtils.clamp(-(endPt.v - pPrev.v) * 18, 0, 1);
            if (slopeDown > 0.32) {
              droppedCount++;
              spawnEdgeBead(
                endPt.u,
                endPt.v - job.brushRadius * 0.55,
                0,
                -0.005,
                0.0019 + currentFogLevel * 0.0005,
                true,
              );
            }
          }
        }
      }
      job.finishHoldSec += dt;
      if (job.finishHoldSec >= 0.18) {
        activeJob = null;
      }
      return;
    }

    // Subtle hairline cursive ligature glide (牽絲) between adjacent Latin lowercase strokes
    if (job.isLigatureJump) {
      job.ligatureTimer += dt;
      const t = THREE.MathUtils.clamp(
        job.ligatureTimer / Math.max(0.008, job.ligatureDurationSec),
        0,
        1,
      );
      const ease = t * t * (3 - 2 * t);
      const nextU = THREE.MathUtils.lerp(job.ligatureStartU, job.ligatureEndU, ease);
      const nextV = THREE.MathUtils.lerp(job.ligatureStartV, job.ligatureEndV, ease);

      paintWipeCircle(nextU, nextV, job.brushRadius * 0.28, 0.24, 0, -1, 0.18);
      job.tipU = nextU;
      job.tipV = nextV;

      if (t >= 1) {
        const nextStroke = job.strokes[job.strokeIndex];
        job.isLigatureJump = false;
        job.targetDist = 0;
        job.strokeElapsed = 0;
        job.tipU = nextStroke.points[0].u;
        job.tipV = nextStroke.points[0].v;
        const entryDir = nextStroke.fieldDirs[0] ?? { u: 0, v: 0 };
        job.tipVx = entryDir.u * job.baseSpeedUvPerSec * 0.22;
        job.tipVy = entryDir.v * job.baseSpeedUvPerSec * 0.22;
        job.strokeStarted = false;
        job.tipHoldSec = 0;
      }
      return;
    }

    if (job.liftCooldownSec > 0) {
      job.liftCooldownSec -= dt;
      if (job.liftCooldownSec <= 0) {
        const nextStroke = job.strokes[job.strokeIndex];
        if (nextStroke.linkToPrev === 'ligature') {
          job.isLigatureJump = true;
          job.ligatureStartU = job.tipU;
          job.ligatureStartV = job.tipV;
          job.ligatureEndU = nextStroke.points[0].u;
          job.ligatureEndV = nextStroke.points[0].v;
          job.ligatureDurationSec = 0.015;
          job.ligatureTimer = 0;
          return;
        }
        job.targetDist = 0;
        job.strokeElapsed = 0;
        job.tipU = nextStroke.points[0].u;
        job.tipV = nextStroke.points[0].v;
        if (nextStroke.linkToPrev === 'flow') {
          const entryDir = nextStroke.fieldDirs[0] ?? { u: 0, v: 0 };
          job.tipVx = entryDir.u * job.baseSpeedUvPerSec * 0.18;
          job.tipVy = entryDir.v * job.baseSpeedUvPerSec * 0.18;
        } else {
          job.tipVx = 0;
          job.tipVy = 0;
        }
        job.strokeStarted = false;
        job.tipHoldSec = 0;
      }
      return;
    }

    const stroke = job.strokes[job.strokeIndex];
    job.strokeElapsed += dt;

    if (!job.strokeStarted) {
      job.strokeStarted = true;
      job.pressure = 0.88;
      paintWipeCircle(
        job.tipU,
        job.tipV,
        job.brushRadius * 0.94,
        0.98,
        0,
        -1,
        job.pressure,
      );
    }

    if (job.tipHoldSec > 0) {
      job.tipHoldSec -= dt;
      if (job.tipHoldSec <= 0) {
        job.strokeIndex += 1;
        const upcoming = job.strokes[job.strokeIndex];
        if (upcoming && upcoming.linkToPrev === 'ligature') {
          job.liftCooldownSec = 0.001;
        } else if (upcoming && upcoming.linkToPrev === 'flow') {
          job.liftCooldownSec = 0.003;
        } else {
          job.liftCooldownSec = stroke.isSmallDetail ? 0.004 : 0.008;
        }
      }
      return;
    }

    const subSteps = Math.max(1, Math.min(4, Math.ceil(dt / 0.008)));
    const subDt = dt / subSteps;

    for (let step = 0; step < subSteps; step++) {
      const target = sampleStrokeTarget(stroke, job.targetDist);
      const localWet = sampleFieldBilinear(wetAmount, job.tipU, job.tipV);
      const localFogDrag = currentFogLevel * (1.0 - localWet * 0.45);
      const dragFactor = 0.86 + localFogDrag * 0.34;

      const lagDist = Math.hypot((target.u - job.tipU) * currentMirrorAspect, target.v - job.tipV);
      const lagThrottle = THREE.MathUtils.clamp(1.20 - lagDist * 22, 0.35, 1.0);
      const curveBrake = THREE.MathUtils.clamp(1.06 - target.curvature * 0.48, 0.56, 1.06);
      const endEase = job.targetDist > stroke.totalLength * 0.88 ? 0.78 : 1.0;

      const advanceSpeed =
        (job.baseSpeedUvPerSec * curveBrake * lagThrottle * endEase) / dragFactor;
      job.targetDist = Math.min(stroke.totalLength, job.targetDist + advanceSpeed * subDt);

      const springK = 1220 / dragFactor;
      const springC = 56 * dragFactor;

      const fieldAssist = (advanceSpeed * 0.25) / Math.max(0.05, dragFactor);
      const stickSlip =
        Math.sin((elapsed + job.phaseSeed) * 28.0 + job.targetDist * 140) *
        0.00018 *
        (0.4 + localFogDrag);

      const ax =
        springK * (target.u - job.tipU) +
        target.dirU * fieldAssist * 10.0 -
        springC * job.tipVx +
        stickSlip;
      const ay =
        springK * (target.v - job.tipV) +
        target.dirV * fieldAssist * 10.0 -
        springC * job.tipVy;

      job.tipVx += ax * subDt;
      job.tipVy += ay * subDt;

      const nextU = THREE.MathUtils.clamp(job.tipU + job.tipVx * subDt, 0.01, 0.99);
      const nextV = THREE.MathUtils.clamp(job.tipV + job.tipVy * subDt, 0.01, 0.99);

      const segDist = Math.hypot((nextU - job.tipU) * currentMirrorAspect, nextV - job.tipV);
      const speedNorm = THREE.MathUtils.clamp(segDist / Math.max(0.003, subDt) / 1.4, 0.18, 2.0);

      const strokeProgress = THREE.MathUtils.clamp(
        job.targetDist / Math.max(0.002, stroke.totalLength),
        0,
        1,
      );
      const rhythmicEnvelope =
        strokeProgress < 0.14
          ? 0.10 * (1 - strokeProgress / 0.14)
          : strokeProgress > 0.84
            ? 0.08 * ((strokeProgress - 0.84) / 0.16)
            : -0.04;
      const downComp = THREE.MathUtils.clamp(-(nextV - job.tipV) / Math.max(1e-4, segDist), 0, 1);
      const targetPressure = THREE.MathUtils.clamp(
        0.52 +
          rhythmicEnvelope +
          (1.0 - Math.min(1, speedNorm * 0.55)) * 0.28 +
          downComp * 0.16 +
          target.curvature * 0.12,
        0.44,
        1.0,
      );
      job.pressure = THREE.MathUtils.lerp(job.pressure, targetPressure, 0.35);

      const endPt = stroke.points[stroke.points.length - 1];
      const distToEnd = Math.hypot(
        (endPt.u - nextU) * currentMirrorAspect,
        endPt.v - nextV,
      );
      const settleDist = Math.max(0.006, job.brushRadius * 1.15);
      const isEnd =
        (job.targetDist >= stroke.totalLength && distToEnd < settleDist) ||
        job.strokeElapsed > 0.55;

      fluidDisplacementStroke(
        job.tipU,
        job.tipV,
        isEnd ? endPt.u : nextU,
        isEnd ? endPt.v : nextV,
        job.brushRadius,
        isEnd ? 0.28 : speedNorm,
        isEnd,
        job.pressure,
        true,
      );

      job.tipU = isEnd ? endPt.u : nextU;
      job.tipV = isEnd ? endPt.v : nextV;

      if (isEnd) {
        job.tipHoldSec = stroke.isSmallDetail ? 0.003 : 0.006;
        break;
      }
    }
  };

  // Surface Tension + Capillary Meandering Rivulets + Plateau-Rayleigh Beading + Gravity Acceleration
  let ambientDripTimer = 0;
  const updateAndRenderBeads = (dt: number) => {
    // Subtle natural ambient condensation rivulets when fog saturates (never crowding user/lyric strokes)
    if (currentFogLevel > 0.68 && activeBeads.length < 90) {
      ambientDripTimer += dt * (0.4 + currentTempDiff * 0.8);
      if (ambientDripTimer > 3.2) {
        ambientDripTimer = 0;
        const ru = 0.08 + Math.random() * 0.84;
        const rv = 0.55 + Math.random() * 0.40;
        spawnEdgeBead(
          ru,
          rv,
          (Math.random() - 0.5) * 0.004,
          -0.018,
          0.0029 + Math.random() * 0.0014,
          true,
        );
      }
    }

    if (activeBeads.length === 0) {
      // Nothing to draw: clear + upload once, then leave the texture alone.
      if (dropletHasContent) {
        dropletCtx.fillStyle = '#000000';
        dropletCtx.fillRect(0, 0, dropDims.w, dropDims.h);
        dropletTexture.needsUpdate = true;
        dropletHasContent = false;
        dropletDirty = true;
      }
      return;
    }

    dropletCtx.fillStyle = '#000000';
    dropletCtx.fillRect(0, 0, dropDims.w, dropDims.h);
    dropletHasContent = true;

    const rCrit = THREE.MathUtils.lerp(0.0040, 0.0025, currentFogLevel) * (0.75 + SURFACE_TENSION * 0.55);
    const gravity = 0.58;
    const cellStep = 1 / WET_GRID_SIZE;

    // 1. Capillary meniscus confluence attraction + volume-conserving coalescence
    // Sweep beads sorted by u so only neighbours inside the maximum interaction reach
    // ((rMax + rMax) * 4.4 = 0.0748 iso units) are compared, instead of every pair.
    const beadCount = activeBeads.length;
    beadOrder.length = beadCount;
    for (let i = 0; i < beadCount; i++) beadOrder[i] = i;
    beadOrder.sort((p, q) => activeBeads[p].u - activeBeads[q].u);
    const maxReachU = 0.075 / Math.max(0.05, currentMirrorAspect);

    for (let oi = 0; oi < beadCount; oi++) {
      const a = activeBeads[beadOrder[oi]];
      if (a.r <= 0) continue;
      for (let oj = oi + 1; oj < beadCount; oj++) {
        const b = activeBeads[beadOrder[oj]];
        if (b.u - a.u > maxReachU) break;
        if (b.r <= 0) continue;
        const dxIso = (a.u - b.u) * currentMirrorAspect;
        const dyIso = a.v - b.v;
        const dist = Math.hypot(dxIso, dyIso);
        const mergeDist = (a.r + b.r) * 1.24;
        const attractDist = (a.r + b.r) * 4.4;

        if (dist < mergeDist) {
          const mA = a.r * a.r * a.r;
          const mB = b.r * b.r * b.r;
          const mSum = mA + mB;
          a.u = (a.u * mA + b.u * mB) / mSum;
          a.v = (a.v * mA + b.v * mB) / mSum;
          a.vx = (a.vx * mA + b.vx * mB) / mSum;
          a.vy = (a.vy * mA + b.vy * mB) / mSum;
          a.r = Math.min(0.0085, Math.cbrt(mSum));
          a.rolling = a.rolling || b.rolling || a.r >= rCrit * 0.92;
          a.maxLife = Math.max(a.maxLife, a.life + 1.6);
          b.r = -1;
        } else if (dist < attractDist && dist > 0.0001) {
          const pull = 0.090 * (1 - dist / attractDist) * dt;
          const nx = dxIso / dist;
          const ny = dyIso / dist;
          if (a.r >= b.r) {
            b.vx += nx * pull;
            b.vy += ny * pull;
          } else {
            a.vx -= nx * pull;
            a.vy -= ny * pull;
          }
        }
      }
    }

    for (let i = activeBeads.length - 1; i >= 0; i--) {
      const b = activeBeads[i];
      if (b.r <= 0) {
        activeBeads.splice(i, 1);
        continue;
      }

      b.life += dt;
      if (b.life >= b.maxLife || b.v < 0.012) {
        activeBeads.splice(i, 1);
        continue;
      }

      const wetHere = sampleFieldBilinear(wetAmount, b.u, b.v);
      const wetE = sampleFieldBilinear(wetAmount, b.u + cellStep, b.v);
      const wetW = sampleFieldBilinear(wetAmount, b.u - cellStep, b.v);
      const wetN = sampleFieldBilinear(wetAmount, b.u, b.v + cellStep);
      const wetS = sampleFieldBilinear(wetAmount, b.u, b.v - cellStep);
      const gradX = (wetE - wetW) * 0.5;
      const gradY = (wetN - wetS) * 0.5;

      const forceHere = sampleFieldBilinear(wetForce, b.u, b.v);
      const dirX = sampleFieldBilinear(wetDirX, b.u, b.v);
      const dirY = sampleFieldBilinear(wetDirY, b.u, b.v);

      // Ambient steam & swept condensation feed droplets as they descend
      const condGrow =
        dt *
        0.00042 *
        currentFogLevel *
        (0.55 + currentTempDiff * 0.65 + forceHere * 0.70);
      b.r = Math.min(0.0085, b.r + condGrow);

      const channelPullX = wetHere > 0.12 ? gradX * 2.1 : 0;
      const channelPullY = wetHere > 0.12 ? Math.min(0, gradY) * 1.4 : 0;

      const pathDownWeight = dirY < -0.15 ? -dirY : 0.25;
      const flowSteerX = dirX * forceHere * 0.22;
      const flowSteerY = -pathDownWeight * forceHere * 0.12;

      if (!b.rolling && (b.r >= rCrit || (forceHere > 0.58 && b.r >= rCrit * 0.84))) {
        b.rolling = true;
        b.vy = -0.020;
      }

      if (b.rolling) {
        const ratioSq = Math.min(0.92, (rCrit / Math.max(0.0008, b.r)) ** 2);
        const gravDrive = Math.max(0.14, 1.0 - ratioSq);
        const stickSlip = 0.60 + 0.40 * Math.sin(b.life * 6.5 + b.slipPhase);
        const wetLubrication = 0.78 + wetHere * 0.75;
        // Organic sinuous capillary meandering (as drops latch onto uneven micro-condensation cells)
        const meander =
          Math.sin(b.v * 44 + b.seed) * 0.024 +
          Math.cos(b.v * 97 + b.seed * 1.9) * 0.015;

        const ay =
          -gravity * gravDrive * wetLubrication * stickSlip +
          channelPullY +
          flowSteerY;
        const ax = channelPullX + flowSteerX + meander;

        const viscousDrag = 2.7 / Math.max(0.28, b.r * 280);
        b.vy = THREE.MathUtils.clamp(
          b.vy + (ay - b.vy * viscousDrag) * dt,
          -0.30,
          0.015,
        );
        b.vx = THREE.MathUtils.clamp(
          b.vx * Math.exp(-4.4 * dt) + ax * dt,
          -0.14,
          0.14,
        );
      } else {
        b.vx = (b.vx + (channelPullX * 0.45 + flowSteerX * 0.65) * dt) * Math.exp(-5.2 * dt);
        b.vy = (b.vy + (flowSteerY * 0.75 - 0.018 * Math.max(0, b.r / rCrit - 0.4)) * dt) * Math.exp(-5.2 * dt);
      }

      b.prevU = b.u;
      b.prevV = b.v;
      b.u = THREE.MathUtils.clamp(
        b.u + (b.vx * dt) / Math.max(0.35, currentMirrorAspect),
        0.01,
        0.99,
      );
      b.v += b.vy * dt;

      if (b.rolling) {
        const fade = 1 - b.life / b.maxLife;
        const speedMag = Math.hypot(b.vx, b.vy);
        if (fade > 0.30 && speedMag > 0.008) {
          // Carve continuous smooth round-capped rivulet path (no dotted bead gaps!)
          const prevX = b.prevU * wipeDims.w;
          const prevY = (1 - b.prevV) * wipeDims.h;
          const wakeCx = b.u * wipeDims.w;
          const wakeCy = (1 - b.v) * wipeDims.h;
          const wakeR = Math.max(1.0, b.r * wipeDims.h * 0.38);

          wipeCtx.save();
          // 'lighten' keeps overlapping round caps from stacking alpha (which looked like dotted beads)
          wipeCtx.globalCompositeOperation = 'lighten';
          wipeCtx.strokeStyle = `rgba(255, 255, 255, ${0.62 * Math.min(1, fade * 1.4)})`;
          wipeCtx.lineWidth = wakeR * 2.0;
          wipeCtx.lineCap = 'round';
          wipeCtx.beginPath();
          wipeCtx.moveTo(prevX, prevY);
          wipeCtx.lineTo(wakeCx, wakeCy);
          wipeCtx.stroke();
          wipeCtx.restore();
          wipeDirty = true;

          // Continuous translucent liquid film + subtle displaced mist sheen along wake
          wetFilmCtx.save();
          wetFilmCtx.globalCompositeOperation = 'lighten';
          wetFilmCtx.strokeStyle = `rgba(240, 120, 0, ${0.34 * Math.min(1, fade * 1.4)})`;
          wetFilmCtx.lineWidth = Math.max(1.4, b.r * wetDims.h * 0.9);
          wetFilmCtx.lineCap = 'round';
          wetFilmCtx.beginPath();
          wetFilmCtx.moveTo(b.prevU * wetDims.w, (1 - b.prevV) * wetDims.h);
          wetFilmCtx.lineTo(b.u * wetDims.w, (1 - b.v) * wetDims.h);
          wetFilmCtx.stroke();
          wetFilmCtx.restore();
          wetFilmDirty = true;

          // Plateau-Rayleigh filament pinch-off: occasional tiny stationary pearl bead inside fast wakes
          if (speedMag > 0.090 && activeBeads.length < 200 && Math.random() < dt * 0.35) {
            spawnEdgeBead(
              b.u + (Math.random() - 0.5) * 0.0016,
              b.v + b.r * 1.4,
              0,
              -0.001,
              0.0010 + Math.random() * 0.0005,
              false,
            );
          }
        }
      }

      const fade = 1 - b.life / b.maxLife;
      const visR = b.r * (0.80 + 0.20 * fade);
      paintDropletOnCleanCanvas(
        b.u,
        b.v,
        visR * 1.18,
        0.96 * Math.min(1, fade * 1.5),
        b.rolling ? 1.38 : 1.06,
      );
    }

    dropletTexture.needsUpdate = true;
    dropletDirty = true;
  };

  return {
    screenMaterial,
    steamParticles,
    fluidInputs: {
      wipeMap: wipeTexture,
      dropletMap: dropletTexture,
      wetFilmMap: wetFilmTexture,
      wipeTexSize: screenMaterial.uniforms.uWipeTexSize.value as THREE.Vector2,
      dropTexSize: screenMaterial.uniforms.uDropTexSize.value as THREE.Vector2,
      wetTexSize: screenMaterial.uniforms.uWetTexSize.value as THREE.Vector2,
    },
    setTextures: (
      sharpTex: THREE.Texture,
      nearBlurTex: THREE.Texture,
      farBlurTex: THREE.Texture,
      fluidLayerTex: THREE.Texture,
      strokeDepthTex: THREE.Texture,
    ) => {
      screenMaterial.uniforms.uSharpTex.value = sharpTex;
      screenMaterial.uniforms.uNearBlurTex.value = nearBlurTex;
      screenMaterial.uniforms.uBlurTex.value = farBlurTex;
      screenMaterial.uniforms.uFluidLayerTex.value = fluidLayerTex;
      screenMaterial.uniforms.uStrokeDepthTex.value = strokeDepthTex;
    },
    setMirrorAspect: (aspect: number) => {
      const clampedAspect = Math.max(0.32, aspect);
      const changed = Math.abs(clampedAspect - currentMirrorAspect) > 0.03;
      currentMirrorAspect = clampedAspect;
      screenMaterial.uniforms.uMirrorAspect.value = clampedAspect;

      if (changed) {
        const nextWipeDims = computeAspectDims(clampedAspect, WIPE_MAX_DIM);
        const nextDropDims = computeAspectDims(clampedAspect, DROPLET_MAX_DIM);
        const nextWetDims = computeAspectDims(clampedAspect, WET_FILM_MAX_DIM);

        if (nextWipeDims.w !== wipeDims.w || nextWipeDims.h !== wipeDims.h) {
          const snapshot = document.createElement('canvas');
          snapshot.width = wipeDims.w;
          snapshot.height = wipeDims.h;
          snapshot.getContext('2d')!.drawImage(wipeCanvas, 0, 0);

          wipeDims = nextWipeDims;
          wipeCanvas.width = wipeDims.w;
          wipeCanvas.height = wipeDims.h;
          wipeCtx.drawImage(snapshot, 0, 0, wipeDims.w, wipeDims.h);
          screenMaterial.uniforms.uWipeTexSize.value.set(wipeDims.w, wipeDims.h);
          wipeTexture.needsUpdate = true;
        }

        if (nextDropDims.w !== dropDims.w || nextDropDims.h !== dropDims.h) {
          dropDims = nextDropDims;
          dropletCanvas.width = dropDims.w;
          dropletCanvas.height = dropDims.h;
          dropletHasContent = true; // resized canvas is blank: force a clear + upload next frame
          fluidHold = 3;
          screenMaterial.uniforms.uDropTexSize.value.set(dropDims.w, dropDims.h);
        }

        if (nextWetDims.w !== wetDims.w || nextWetDims.h !== wetDims.h) {
          const snapshotWet = document.createElement('canvas');
          snapshotWet.width = wetDims.w;
          snapshotWet.height = wetDims.h;
          snapshotWet.getContext('2d')!.drawImage(wetFilmCanvas, 0, 0);

          wetDims = nextWetDims;
          wetFilmCanvas.width = wetDims.w;
          wetFilmCanvas.height = wetDims.h;
          wetFilmCtx.drawImage(snapshotWet, 0, 0, wetDims.w, wetDims.h);
          screenMaterial.uniforms.uWetTexSize.value.set(wetDims.w, wetDims.h);
          wetFilmTexture.needsUpdate = true;
        }
      }
    },
    triggerBeatPulse: (intensity = 0.7) => {
      beatPulse = Math.max(beatPulse, THREE.MathUtils.clamp(intensity, 0.15, 1));
    },
    wipeAtUv: (u: number, v: number, radiusNorm?: number) => {
      const rad = radiusNorm ?? strokeWeightToFingerRadius(currentStrokeWeight);
      paintWipeCircle(u, v, rad, 0.96, 0, -1, 0.55);
      if (currentFogLevel > 0.25) {
        const rimR = rad * 1.10;
        const beadCount = currentStrokeWeight < 0.35 ? 1 : 2;
        for (let k = 0; k < beadCount; k++) {
          const ang = -Math.PI * 0.25 - Math.random() * Math.PI * 0.5;
          const bu = u + (Math.cos(ang) * rimR) / Math.max(0.35, currentMirrorAspect);
          const bv = v + Math.sin(ang) * rimR;
          spawnEdgeBead(
            bu,
            bv,
            Math.cos(ang) * 0.008,
            -0.006,
            0.0017 + currentStrokeWeight * 0.0012,
            false,
          );
        }
      }
    },
    wipeStroke: (
      u0: number,
      v0: number,
      u1: number,
      v1: number,
      dtSec = 0.016,
    ) => {
      const dxIso = (u1 - u0) * currentMirrorAspect;
      const dyIso = v1 - v0;
      const dist = Math.hypot(dxIso, dyIso);
      if (dist < 0.0002) return;

      const safeDt = THREE.MathUtils.clamp(dtSec, 0.004, 0.12);
      const speed = dist / safeDt;
      const speedNorm = THREE.MathUtils.clamp(speed / 1.85, 0.08, 2.2);

      const userRadius = strokeWeightToFingerRadius(currentStrokeWeight);
      const strokeRadius =
        userRadius * THREE.MathUtils.lerp(1.05, 0.85, Math.min(1, speedNorm * 0.55));

      fluidDisplacementStroke(u0, v0, u1, v1, strokeRadius, speedNorm, false, 0.62);
    },
    writeLyricOnGlass: (text: string, durationSec = 0.9, isChorus = false) => {
      const clean = text.trim();
      if (!clean) return;
      const seq = ++writeSequence;
      void (async () => {
        // Stroke data first: Latin / kana / hangul are built in, CJK comes from the server.
        await preloadStrokeMediansForText(clean);
        // Web fonts are only needed for characters that have no stroke data (rare fallback),
        // so most lyric lines no longer wait up to 900ms for font loading.
        const missing = Array.from(clean)
          .filter((ch) => ch.trim() && !getTrueStrokeOrderGlyph(ch))
          .join('');
        if (missing) await ensureGlyphFontsReady(missing, 72);
        if (seq !== writeSequence) return;
        prepareHandwritingJob(clean, durationSec, isChorus);
      })();
    },
    triggerSteamBurst: () => {
      manualClearHold = false;
      burstBoost = 0.95;
    },
    isFluidActive: () => fluidHold > 0,
    clearAllFog: () => {
      manualClearHold = true;
      currentFogLevel = 0;
      burstBoost = 0;
      activeJob = null;
      activeBeads.length = 0;
      wetAmount.fill(0);
      wetDirX.fill(0);
      wetDirY.fill(-1);
      wetForce.fill(0);
      wipeCtx.fillStyle = '#000000';
      wipeCtx.fillRect(0, 0, wipeDims.w, wipeDims.h);
      wipeTexture.needsUpdate = true;
      dropletCtx.fillStyle = '#000000';
      dropletCtx.fillRect(0, 0, dropDims.w, dropDims.h);
      dropletTexture.needsUpdate = true;
      dropletHasContent = false;
      fluidHold = 3;
      // Leave faint residual dampness when steam clears rather than dead flat glass
      wetFilmCtx.fillStyle = 'rgba(0, 0, 0, 0.65)';
      wetFilmCtx.fillRect(0, 0, wetDims.w, wetDims.h);
      wetFilmTexture.needsUpdate = true;
      approximateWipedRatio = 0;
    },
    update: (
      dt: number,
      elapsed: number,
      breathCycle: boolean,
      recondenseSpeed: number,
      maxDensity: number,
      fogCoverage: number,
      tempDiff: number,
      audioEnergy = 0,
      lyricsActive = false,
      handwritingScale = 1.05,
      strokeWeight = 0.36,
      parallaxX = 0,
      parallaxY = 0,
    ): FogStatus => {
      const clampedDt = Math.min(dt, 0.1);
      currentTempDiff = tempDiff;
      currentHandwritingScale = handwritingScale;
      currentStrokeWeight = strokeWeight;
      let phase: FogPhase = 'saturated';

      advanceHandwritingJob(clampedDt, elapsed);
      updateAndRenderBeads(clampedDt);

      beatPulse = THREE.MathUtils.damp(beatPulse, 0, 5.5, clampedDt);
      const thermalSpeedMult = 0.85 + tempDiff * 0.4;

      if (manualClearHold && !lyricsActive) {
        currentFogLevel = THREE.MathUtils.damp(currentFogLevel, 0, 10, clampedDt);
        phase = 'clear';
      } else if (breathCycle && !lyricsActive) {
        breathTimer += clampedDt * recondenseSpeed * thermalSpeedMult * 0.75;
        const wave = 0.5 + 0.5 * Math.sin(breathTimer);
        const target = THREE.MathUtils.lerp(maxDensity * 0.52, maxDensity, wave);
        currentFogLevel = THREE.MathUtils.damp(currentFogLevel, target, 4.5, clampedDt);
        phase = wave > 0.5 ? 'fogging' : 'clearing';
      } else {
        const audioLift = lyricsActive ? audioEnergy * 0.12 : 0;
        let target = Math.min(1, maxDensity + audioLift);
        if (burstBoost > 0.005) {
          target = Math.min(1, Math.max(target, maxDensity) + burstBoost * 0.2);
          burstBoost = THREE.MathUtils.damp(burstBoost, 0, 2.4, clampedDt);
        }
        const condRate = 3.2 + recondenseSpeed * 2.2 + burstBoost * 6.0;
        currentFogLevel = THREE.MathUtils.damp(
          currentFogLevel,
          target,
          condRate,
          clampedDt,
        );
        phase = currentFogLevel < maxDensity * 0.88 ? 'fogging' : 'saturated';
      }

      wipeRecoverTimer += clampedDt * (0.48 + tempDiff * 0.42) * recondenseSpeed;
      if (wipeRecoverTimer > 0.22) {
        if (currentFogLevel > 0.15) {
          // Gentle slow re-condensation (~9-14s clear readable persistence before mist fills back in)
          const stepFade = activeJob ? 0.005 : 0.011;
          if (FADE_FLOOR_FIX) {
            wipeFadeDebt += stepFade;
            if (wipeFadeDebt >= MIN_WIPE_FADE_ALPHA) {
              wipeCtx.fillStyle = `rgba(0, 0, 0, ${wipeFadeDebt})`;
              wipeCtx.fillRect(0, 0, wipeDims.w, wipeDims.h);
              wipeFadeDebt = 0;
              wipeDirty = true;
            }
          } else {
            wipeCtx.fillStyle = `rgba(0, 0, 0, ${stepFade})`;
            wipeCtx.fillRect(0, 0, wipeDims.w, wipeDims.h);
            wipeDirty = true;
          }
          const decay = 1 - stepFade * 1.15;
          for (let i = 0; i < wetAmount.length; i++) {
            wetAmount[i] *= decay;
            wetForce[i] *= decay;
          }
          approximateWipedRatio = Math.max(
            0,
            approximateWipedRatio - stepFade * 0.5,
          );
        }

        // Liquid water membrane persists longer on clear glass than white mist, decaying gently via evaporation
        if (FADE_FLOOR_FIX) {
          wetFadeDebt += 0.007;
          if (wetFadeDebt >= MIN_WET_FADE_ALPHA) {
            wetFilmCtx.fillStyle = `rgba(0, 0, 0, ${wetFadeDebt})`;
            wetFilmCtx.fillRect(0, 0, wetDims.w, wetDims.h);
            wetFadeDebt = 0;
            wetFilmDirty = true;
          }
        } else {
          wetFilmCtx.fillStyle = 'rgba(0, 0, 0, 0.007)';
          wetFilmCtx.fillRect(0, 0, wetDims.w, wetDims.h);
          wetFilmDirty = true;
        }
        wipeRecoverTimer = 0;
      }

      // Tell the renderer whether the fluid-normal / stroke-depth inputs changed recently
      if (wipeDirty || wetFilmDirty || dropletDirty) {
        fluidHold = 3;
      } else if (fluidHold > 0) {
        fluidHold--;
      }
      dropletDirty = false;

      if (wipeDirty) {
        wipeTexture.needsUpdate = true;
        wipeDirty = false;
      }
      if (wetFilmDirty) {
        wetFilmTexture.needsUpdate = true;
        wetFilmDirty = false;
      }

      screenMaterial.uniforms.uTime.value = elapsed;
      screenMaterial.uniforms.uFogLevel.value = currentFogLevel;
      screenMaterial.uniforms.uAudioEnergy.value = audioEnergy;
      screenMaterial.uniforms.uBeatPulse.value = beatPulse;
      screenMaterial.uniforms.uParallax.value.set(
        THREE.MathUtils.damp(screenMaterial.uniforms.uParallax.value.x, parallaxX, 6.0, clampedDt),
        THREE.MathUtils.damp(screenMaterial.uniforms.uParallax.value.y, parallaxY, 6.0, clampedDt),
      );
      screenMaterial.uniforms.uCoverage.value = THREE.MathUtils.damp(
        screenMaterial.uniforms.uCoverage.value,
        fogCoverage,
        8.0,
        clampedDt,
      );
      screenMaterial.uniforms.uTempDiff.value = THREE.MathUtils.damp(
        screenMaterial.uniforms.uTempDiff.value,
        tempDiff,
        6.0,
        clampedDt,
      );

      const posAttr = steamGeo.attributes.position as THREE.BufferAttribute;
      const arr = posAttr.array as Float32Array;
      const steamBoost =
        1 + audioEnergy * 1.45 + beatPulse * 0.9 + (recondenseSpeed - 1) * 0.35;
      for (let i = 0; i < particleCount; i++) {
        arr[i * 3] +=
          (velocities[i * 2] + Math.sin(elapsed * 1.3 + phases[i]) * 0.025) *
          clampedDt;
        arr[i * 3 + 1] += velocities[i * 2 + 1] * steamBoost * clampedDt;

        if (arr[i * 3 + 1] > 1.15) {
          arr[i * 3] = (Math.random() - 0.5) * 2.1;
          arr[i * 3 + 1] = -1.15;
        }
      }
      posAttr.needsUpdate = true;

      const steamVisibility = THREE.MathUtils.clamp(
        0.04 + currentFogLevel * 0.09 + audioEnergy * 0.07,
        0,
        0.2,
      );
      steamMat.opacity = THREE.MathUtils.damp(
        steamMat.opacity,
        steamVisibility,
        6.0,
        clampedDt,
      );

      return {
        fogLevel: currentFogLevel,
        phase,
        wipedRatio: approximateWipedRatio,
      };
    },
    dispose: () => {
      screenMaterial.dispose();
      wipeTexture.dispose();
      dropletTexture.dispose();
      wetFilmTexture.dispose();
      steamGeo.dispose();
      steamMat.dispose();
      steamTex.dispose();
    },
  };
}

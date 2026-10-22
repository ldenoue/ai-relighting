import { NeuralUpscaler, type NeuralUpscaleMethod } from "./neural-upscaler";

export type UpscaleMethod = "bilinear" | "lanczos2" | "fsr1" | "anime4k" | "websr" | "fsrcnn";
type SpatialUpscaleMethod = Exclude<UpscaleMethod, NeuralUpscaleMethod>;

export type Look = {
  autoAdjust: boolean;
  targetLuma: number;
  manualEv: number;
  backgroundEv: number;
  warmth: number;
  contrast: number;
  isolate: boolean;
  matteThreshold: number;
  labTransfer: boolean;
  referenceStrength: number;
  referencePersonMean: [number, number, number];
  referencePersonStd: [number, number, number];
  referenceBackgroundMean: [number, number, number];
  referenceBackgroundStd: [number, number, number];
  referencePersonLabMean: [number, number, number];
  referencePersonLabStd: [number, number, number];
  referenceBackgroundLabMean: [number, number, number];
  referenceBackgroundLabStd: [number, number, number];
  referenceBackgroundColor: [number, number, number];
  replaceBackground: boolean;
  rimIntensity: number;
  showLight: boolean;
  lightPosition: [number, number];
  audioReactive: boolean;
  audioLevel: number;
  audioBeat: number;
  rimColor: [number, number, number];
  upscaleMethod: UpscaleMethod;
  rcasEnabled: boolean;
};

const vertexShader = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_position.x * .5 + .5, 1. - (a_position.y * .5 + .5));
  gl_Position = vec4(a_position, 0., 1.);
}`;

export const upscaleVertexShader = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * .5 + .5;
  gl_Position = vec4(a_position, 0., 1.);
}`;

const fragmentShader = `#version 300 es
precision highp float;
uniform sampler2D u_video;
uniform sampler2D u_mask;
uniform float u_autoEv;
uniform float u_manualEv;
uniform float u_backgroundEv;
uniform float u_warmth;
uniform float u_contrast;
uniform float u_isolate;
uniform float u_matteThreshold;
uniform float u_labTransfer;
uniform float u_referenceStrength;
uniform vec3 u_sourcePersonMean;
uniform vec3 u_sourcePersonStd;
uniform vec3 u_sourceBackgroundMean;
uniform vec3 u_sourceBackgroundStd;
uniform vec3 u_referencePersonMean;
uniform vec3 u_referencePersonStd;
uniform vec3 u_referenceBackgroundMean;
uniform vec3 u_referenceBackgroundStd;
uniform vec3 u_sourcePersonLabMean;
uniform vec3 u_sourcePersonLabStd;
uniform vec3 u_sourceBackgroundLabMean;
uniform vec3 u_sourceBackgroundLabStd;
uniform vec3 u_referencePersonLabMean;
uniform vec3 u_referencePersonLabStd;
uniform vec3 u_referenceBackgroundLabMean;
uniform vec3 u_referenceBackgroundLabStd;
uniform vec3 u_referenceBackgroundColor;
uniform float u_replaceBackground;
uniform vec2 u_maskTexel;
uniform float u_rimIntensity;
uniform float u_showLight;
uniform vec2 u_lightPosition;
uniform vec3 u_rimColor;
uniform float u_viewZoom;
uniform vec2 u_viewCenter;
in vec2 v_uv;
out vec4 outColor;

vec3 grade(vec3 c, float ev) {
  c *= exp2(ev);
  c.r *= 1. + u_warmth * .16;
  c.b *= 1. - u_warmth * .13;
  c = (c - .5) * u_contrast + .5;
  return c / (1. + max(c - 1., 0.));
}

vec3 transferLook(vec3 c, vec3 sourceMean, vec3 sourceStd, vec3 targetMean, vec3 targetStd, float preserveLuma) {
  vec3 normalized = (c - sourceMean) / max(sourceStd, vec3(.06));
  vec3 matched = normalized * clamp(targetStd, vec3(.06), vec3(.38)) + targetMean;
  // Keep the transfer photographic and avoid hard channel clipping.
  matched = clamp(matched, 0., 1.);
  float oldLuma = dot(c, vec3(.2126, .7152, .0722));
  float newLuma = dot(matched, vec3(.2126, .7152, .0722));
  matched *= mix(1., oldLuma / max(newLuma, .03), preserveLuma);
  return mix(c, matched, u_referenceStrength);
}

float labCurve(float value) {
  return value > .008856 ? pow(value, 1. / 3.) : 7.787 * value + 16. / 116.;
}

float inverseLabCurve(float value) {
  float cubed = value * value * value;
  return cubed > .008856 ? cubed : (value - 16. / 116.) / 7.787;
}

vec3 rgbToLab(vec3 rgb) {
  vec3 linearRgb = mix(rgb / 12.92, pow((rgb + .055) / 1.055, vec3(2.4)), step(vec3(.04045), rgb));
  vec3 xyz = vec3(
    dot(linearRgb, vec3(.4124564, .3575761, .1804375)) / .95047,
    dot(linearRgb, vec3(.2126729, .7151522, .072175)),
    dot(linearRgb, vec3(.0193339, .119192, .9503041)) / 1.08883
  );
  float fx = labCurve(xyz.x);
  float fy = labCurve(xyz.y);
  float fz = labCurve(xyz.z);
  return vec3((116. * fy - 16.) / 100., 500. * (fx - fy) / 128., 200. * (fy - fz) / 128.);
}

vec3 labToRgb(vec3 lab) {
  float fy = (lab.x * 100. + 16.) / 116.;
  float fx = fy + lab.y * 128. / 500.;
  float fz = fy - lab.z * 128. / 200.;
  vec3 xyz = vec3(inverseLabCurve(fx) * .95047, inverseLabCurve(fy), inverseLabCurve(fz) * 1.08883);
  vec3 linearRgb = vec3(
    dot(xyz, vec3(3.2404542, -1.5371385, -.4985314)),
    dot(xyz, vec3(-.969266, 1.8760108, .041556)),
    dot(xyz, vec3(.0556434, -.2040259, 1.0572252))
  );
  vec3 positive = max(linearRgb, vec3(0.));
  return clamp(mix(12.92 * positive, 1.055 * pow(positive, vec3(1. / 2.4)) - .055, step(vec3(.0031308), positive)), 0., 1.);
}

vec3 transferLab(vec3 c, vec3 sourceMean, vec3 sourceStd, vec3 targetMean, vec3 targetStd) {
  vec3 lab = rgbToLab(c);
  vec3 matched = (lab - sourceMean) / max(sourceStd, vec3(.025))
    * clamp(targetStd, vec3(.025), vec3(.65)) + targetMean;
  return labToRgb(matched);
}

vec3 screenLight(vec3 base, vec3 light) {
  return 1. - (1. - base) * (1. - clamp(light, 0., 1.));
}

vec3 srgbToLinear(vec3 color) {
  return mix(color / 12.92, pow((color + .055) / 1.055, vec3(2.4)), step(vec3(.04045), color));
}

vec3 linearToSrgb(vec3 color) {
  vec3 positive = max(color, vec3(0.));
  return clamp(mix(12.92 * positive, 1.055 * pow(positive, vec3(1. / 2.4)) - .055, step(vec3(.0031308), positive)), 0., 1.);
}

vec3 acesToneMap(vec3 color) {
  return clamp((color * (2.51 * color + .03)) / (color * (2.43 * color + .59) + .14), 0., 1.);
}

void main() {
  vec2 sampleUv = (v_uv - vec2(.5)) / u_viewZoom + u_viewCenter;
  float insideView = step(0., sampleUv.x) * step(sampleUv.x, 1.) * step(0., sampleUv.y) * step(sampleUv.y, 1.);
  vec3 source = texture(u_video, clamp(sampleUv, 0., 1.)).rgb * insideView;
  vec2 d = u_maskTexel;
  float softMask = texture(u_mask, sampleUv).r * 4.;
  softMask += (texture(u_mask, sampleUv + vec2(d.x, 0.)).r + texture(u_mask, sampleUv - vec2(d.x, 0.)).r
    + texture(u_mask, sampleUv + vec2(0., d.y)).r + texture(u_mask, sampleUv - vec2(0., d.y)).r) * 2.;
  softMask += texture(u_mask, sampleUv + d).r + texture(u_mask, sampleUv - d).r
    + texture(u_mask, sampleUv + vec2(d.x, -d.y)).r + texture(u_mask, sampleUv + vec2(-d.x, d.y)).r;
  softMask = softMask / 16. * insideView;
  // Keep MediaPipe's confidence continuous. This is both the composite alpha
  // and the optical occlusion term: opaque pixels block the source, while
  // uncertain hair/edge pixels transmit part of the light behind them.
  float matteAlpha = clamp(softMask, 0., 1.);
  // Turn model confidence into a useful but still feathered person alpha.
  // Lower thresholds recover conservative mattes such as RVM; the fixed
  // transition width avoids the hard cardboard edge of a binary cutoff.
  float person = mix(1., smoothstep(u_matteThreshold, min(.99, u_matteThreshold + .35), matteAlpha), u_isolate);
  vec3 foregroundRgb = transferLook(source, u_sourcePersonMean, u_sourcePersonStd, u_referencePersonMean, u_referencePersonStd, 1.);
  vec3 backgroundRgb = transferLook(source, u_sourceBackgroundMean, u_sourceBackgroundStd, u_referenceBackgroundMean, u_referenceBackgroundStd, 0.);
  float labAmount = u_labTransfer * step(.001, u_referenceStrength);
  vec3 foreground = mix(foregroundRgb, transferLab(source, u_sourcePersonLabMean, u_sourcePersonLabStd, u_referencePersonLabMean, u_referencePersonLabStd), labAmount);
  vec3 background = mix(backgroundRgb, transferLab(source, u_sourceBackgroundLabMean, u_sourceBackgroundLabStd, u_referenceBackgroundLabMean, u_referenceBackgroundLabStd), labAmount);
  foreground = grade(foreground, u_autoEv + u_manualEv);

  // Estimate the silhouette normal from the continuous matte. Keeping this
  // radius close to one model texel avoids turning the 256px contour into a
  // thick stroke after the frame is reconstructed at 640px.
  vec2 edgeRadius = d * .9;
  float sampleL = texture(u_mask, sampleUv - vec2(edgeRadius.x, 0.)).r;
  float sampleR = texture(u_mask, sampleUv + vec2(edgeRadius.x, 0.)).r;
  float sampleU = texture(u_mask, sampleUv - vec2(0., edgeRadius.y)).r;
  float sampleD = texture(u_mask, sampleUv + vec2(0., edgeRadius.y)).r;
  float sampleUL = texture(u_mask, sampleUv - edgeRadius).r;
  float sampleDR = texture(u_mask, sampleUv + edgeRadius).r;
  float sampleUR = texture(u_mask, sampleUv + vec2(edgeRadius.x, -edgeRadius.y)).r;
  float sampleDL = texture(u_mask, sampleUv + vec2(-edgeRadius.x, edgeRadius.y)).r;
  float straightGradient = length(vec2(sampleR - sampleL, sampleD - sampleU));
  float diagonalGradient = length(vec2(sampleDR - sampleUL, sampleDL - sampleUR));
  float contourGradient = clamp(straightGradient * .72 + diagonalGradient * .34, 0., 1.);
  float confidenceEdge = 1. - abs(matteAlpha * 2. - 1.);
  float edgeProfile = pow(clamp(contourGradient * .78 + confidenceEdge * .22, 0., 1.), 1.45);
  float innerBand = edgeProfile * smoothstep(.2, .72, matteAlpha);

  vec2 spillRadius = d * 2.75;
  float spillL = texture(u_mask, sampleUv - vec2(spillRadius.x, 0.)).r;
  float spillR = texture(u_mask, sampleUv + vec2(spillRadius.x, 0.)).r;
  float spillU = texture(u_mask, sampleUv - vec2(0., spillRadius.y)).r;
  float spillD = texture(u_mask, sampleUv + vec2(0., spillRadius.y)).r;
  float spillUL = texture(u_mask, sampleUv - spillRadius).r;
  float spillDR = texture(u_mask, sampleUv + spillRadius).r;
  float spillUR = texture(u_mask, sampleUv + vec2(spillRadius.x, -spillRadius.y)).r;
  float spillDL = texture(u_mask, sampleUv + vec2(-spillRadius.x, spillRadius.y)).r;
  float erodedMask = min(min(min(spillL, spillR), min(spillU, spillD)), min(min(spillUL, spillDR), min(spillUR, spillDL)));
  float inwardDistance = max(matteAlpha - erodedMask, 0.);
  float inwardBand = smoothstep(.12, .72, inwardDistance) * smoothstep(.58, .9, matteAlpha) * (1. - innerBand * .72);

  // The light source is a radial field in the background, not a dilation of
  // the silhouette. Auto-framing keeps the upper torso close to this fixed
  // position. Multiple falloffs emulate a hot source plus optical bloom.
  vec2 lightCenter = u_lightPosition;
  vec2 wideDelta = (v_uv - lightCenter) / vec2(.48, .42);
  vec2 coreDelta = (v_uv - lightCenter) / vec2(.22, .2);
  float wideHalo = exp(-dot(wideDelta, wideDelta) * 1.35);
  float mediumHalo = exp(-dot(wideDelta, wideDelta) * 4.8);
  float hotCore = exp(-dot(coreDelta, coreDelta) * 3.2);
  float irradiance = clamp(wideHalo * .12 + mediumHalo * .28 + hotCore * .72, 0., 1.);

  // A backlight only catches contour points whose outward matte normal faces
  // away from the centered source. This prevents every mask edge from being
  // painted with the same opacity.
  vec2 matteGradient = vec2(sampleR - sampleL, sampleD - sampleU);
  float gradientLength = length(matteGradient);
  vec2 outwardNormal = -matteGradient / max(gradientLength, .0001);
  vec2 radialDirection = normalize(v_uv - lightCenter + vec2(.00001));
  float facingAlignment = smoothstep(-.35, .85, dot(outwardNormal, radialDirection));
  float sourceFacing = mix(.18, 1., facingAlignment) * smoothstep(.01, .13, gradientLength);
  float rimIrradiance = mix(.16, 1., smoothstep(.025, .78, irradiance));

  vec2 floorDelta = (v_uv - vec2(.5, .92)) / vec2(.3, .055);
  float floorPool = exp(-dot(floorDelta, floorDelta) * 2.2) * .2;

  vec3 rimColor = u_rimColor;
  float sourceLuma = dot(foreground, vec3(.2126, .7152, .0722));
  float materialResponse = mix(1., .68, sourceLuma);
  float rimShape = innerBand * sourceFacing * rimIrradiance * materialResponse;
  float spillShape = inwardBand * sourceFacing * rimIrradiance * materialResponse;

  // Relight the existing material in linear RGB instead of screening a flat
  // colored strip over it. The high end of the slider reaches a much stronger
  // exposure, while ACES compression retains folds instead of clipping them
  // into a solid white or colored band.
  float intensityCurve = pow(u_rimIntensity, 1.25);
  float rimEnergy = clamp((rimShape + spillShape * .22) * intensityCurve * u_isolate, 0., 1.);
  float exposureStops = mix(1.25, 3.8, pow(u_rimIntensity, 1.6));
  vec3 foregroundLinear = srgbToLinear(clamp(foreground, 0., 1.));
  vec3 lightLinear = srgbToLinear(clamp(rimColor, 0., 1.));
  vec3 litMaterial = foregroundLinear * exp2(exposureStops);
  litMaterial += lightLinear * mix(.12, .045, sourceLuma);
  litMaterial = acesToneMap(litMaterial);
  foregroundLinear = mix(foregroundLinear, litMaterial, min(1., rimEnergy * 1.45));
  foreground = linearToSrgb(foregroundLinear);

  background = grade(background, u_backgroundEv);
  vec3 solidBackground = clamp(u_referenceBackgroundColor * exp2(u_backgroundEv), 0., 1.);
  background = mix(background, solidBackground, u_replaceBackground);
  vec3 behindLight = rimColor * (irradiance + floorPool) * u_rimIntensity * u_isolate * u_showLight;
  background = screenLight(background, behindLight);

  // Compositing with the continuous alpha performs the occlusion: solid body
  // pixels hide the source, while fractional edge/hair pixels reveal a
  // proportional amount of the halo and receive stronger scattering.
  vec3 composite = mix(background, foreground, person);
  outColor = vec4(clamp(composite, 0., 1.), 1.);
}`;

export const bilinearShader = `#version 300 es
precision highp float;
uniform sampler2D u_src;
in vec2 v_uv;
out vec4 outColor;
void main() { outColor = vec4(texture(u_src, v_uv).rgb, 1.); }
`;

export const lanczos2Shader = `#version 300 es
precision highp float;
uniform sampler2D u_src;
uniform vec2 u_inSize;
uniform vec2 u_outSize;
out vec4 outColor;

float sinc(float x) {
  x = abs(x);
  if (x < .0001) return 1.;
  float p = 3.14159265359 * x;
  return sin(p) / p;
}
float lanczos2(float x) {
  return abs(x) < 2. ? sinc(x) * sinc(x * .5) : 0.;
}
vec3 fetchSource(ivec2 p) {
  return texelFetch(u_src, clamp(p, ivec2(0), ivec2(u_inSize) - 1), 0).rgb;
}
void main() {
  vec2 sourcePosition = gl_FragCoord.xy * (u_inSize / u_outSize) - .5;
  ivec2 base = ivec2(floor(sourcePosition));
  vec2 fraction = sourcePosition - vec2(base);
  vec3 color = vec3(0.);
  float weightSum = 0.;
  for (int y = -1; y <= 2; y++) {
    for (int x = -1; x <= 2; x++) {
      float weight = lanczos2(float(x) - fraction.x) * lanczos2(float(y) - fraction.y);
      color += fetchSource(base + ivec2(x, y)) * weight;
      weightSum += weight;
    }
  }
  outColor = vec4(clamp(color / max(weightSum, .0001), 0., 1.), 1.);
}`;

// AMD FidelityFX Super Resolution 1.0 EASU, adapted from the WebGL2
// non-gather port in natureglass/Fsrlab. Its twelve-tap kernel follows local
// edge direction, then clamps to the nearest four texels to suppress ringing.
export const fsrEasuShader = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_src;
uniform vec2 u_inSize;
uniform vec2 u_outSize;
out vec4 outColor;

vec3 fetchSource(ivec2 p) {
  return texelFetch(u_src, clamp(p, ivec2(0), ivec2(u_inSize) - 1), 0).rgb;
}
float luminance(vec3 c) { return c.g + .5 * (c.r + c.b); }
void setDirection(inout vec2 direction, inout float lengthValue, float weight,
  float a, float b, float c, float d, float e) {
  float dc = d - c;
  float cb = c - b;
  float lengthX = 1. / max(max(abs(dc), abs(cb)), 1e-4);
  float directionX = d - b;
  direction.x += directionX * weight;
  lengthX = clamp(abs(directionX) * lengthX, 0., 1.);
  lengthValue += lengthX * lengthX * weight;
  float ec = e - c;
  float ca = c - a;
  float lengthY = 1. / max(max(abs(ec), abs(ca)), 1e-4);
  float directionY = e - a;
  direction.y += directionY * weight;
  lengthY = clamp(abs(directionY) * lengthY, 0., 1.);
  lengthValue += lengthY * lengthY * weight;
}
void easuTap(inout vec3 color, inout float weightSum, vec2 offset,
  vec2 direction, vec2 length2, float lobe, float clipValue, vec3 sampleColor) {
  vec2 vectorValue = vec2(offset.x * direction.x + offset.y * direction.y,
    offset.x * -direction.y + offset.y * direction.x) * length2;
  float distance2 = min(dot(vectorValue, vectorValue), clipValue);
  float windowWeight = .4 * distance2 - 1.;
  float lobeWeight = lobe * distance2 - 1.;
  windowWeight *= windowWeight;
  lobeWeight *= lobeWeight;
  windowWeight = 1.5625 * windowWeight - .5625;
  float weight = windowWeight * lobeWeight;
  color += sampleColor * weight;
  weightSum += weight;
}
void main() {
  vec2 ratio = u_inSize / u_outSize;
  vec2 position = floor(gl_FragCoord.xy) * ratio + .5 * ratio - .5;
  vec2 floorPosition = floor(position);
  position -= floorPosition;
  ivec2 f = ivec2(floorPosition);

  vec3 bC = fetchSource(f + ivec2( 0,-1)), cC = fetchSource(f + ivec2( 1,-1));
  vec3 eC = fetchSource(f + ivec2(-1, 0)), fC = fetchSource(f);
  vec3 gC = fetchSource(f + ivec2( 1, 0)), hC = fetchSource(f + ivec2( 2, 0));
  vec3 iC = fetchSource(f + ivec2(-1, 1)), jC = fetchSource(f + ivec2( 0, 1));
  vec3 kC = fetchSource(f + ivec2( 1, 1)), lC = fetchSource(f + ivec2( 2, 1));
  vec3 nC = fetchSource(f + ivec2( 0, 2)), oC = fetchSource(f + ivec2( 1, 2));
  float bL=luminance(bC), cL=luminance(cC), eL=luminance(eC), fL=luminance(fC);
  float gL=luminance(gC), hL=luminance(hC), iL=luminance(iC), jL=luminance(jC);
  float kL=luminance(kC), lL=luminance(lC), nL=luminance(nC), oL=luminance(oC);

  vec2 direction = vec2(0.);
  float lengthValue = 0.;
  setDirection(direction,lengthValue,(1.-position.x)*(1.-position.y),bL,eL,fL,gL,jL);
  setDirection(direction,lengthValue,position.x*(1.-position.y),cL,fL,gL,hL,kL);
  setDirection(direction,lengthValue,(1.-position.x)*position.y,fL,iL,jL,kL,nL);
  setDirection(direction,lengthValue,position.x*position.y,gL,jL,kL,lL,oL);
  float directionLength = dot(direction, direction);
  bool zeroDirection = directionLength < (1. / 32768.);
  float inverseDirectionLength = inversesqrt(max(directionLength, 1e-12));
  direction.x = zeroDirection ? 1. : direction.x;
  direction *= zeroDirection ? 1. : inverseDirectionLength;
  lengthValue = lengthValue * .5;
  lengthValue *= lengthValue;
  float stretch = dot(direction,direction) / max(max(abs(direction.x),abs(direction.y)),1e-5);
  vec2 length2 = vec2(1. + (stretch - 1.) * lengthValue, 1. - .5 * lengthValue);
  float lobe = .5 + ((.25 - .04) - .5) * lengthValue;
  float clipValue = 1. / lobe;
  vec3 minimum4 = min(min(fC,gC),min(jC,kC));
  vec3 maximum4 = max(max(fC,gC),max(jC,kC));
  vec3 accumulated = vec3(0.);
  float weightSum = 0.;
  easuTap(accumulated,weightSum,vec2( 0.,-1.)-position,direction,length2,lobe,clipValue,bC);
  easuTap(accumulated,weightSum,vec2( 1.,-1.)-position,direction,length2,lobe,clipValue,cC);
  easuTap(accumulated,weightSum,vec2(-1., 1.)-position,direction,length2,lobe,clipValue,iC);
  easuTap(accumulated,weightSum,vec2( 0., 1.)-position,direction,length2,lobe,clipValue,jC);
  easuTap(accumulated,weightSum,vec2( 0., 0.)-position,direction,length2,lobe,clipValue,fC);
  easuTap(accumulated,weightSum,vec2( 1., 0.)-position,direction,length2,lobe,clipValue,gC);
  easuTap(accumulated,weightSum,vec2( 1., 1.)-position,direction,length2,lobe,clipValue,kC);
  easuTap(accumulated,weightSum,vec2( 2., 1.)-position,direction,length2,lobe,clipValue,lC);
  easuTap(accumulated,weightSum,vec2( 2., 0.)-position,direction,length2,lobe,clipValue,hC);
  easuTap(accumulated,weightSum,vec2(-1., 0.)-position,direction,length2,lobe,clipValue,eC);
  easuTap(accumulated,weightSum,vec2( 0., 2.)-position,direction,length2,lobe,clipValue,nC);
  easuTap(accumulated,weightSum,vec2( 1., 2.)-position,direction,length2,lobe,clipValue,oC);
  outColor = vec4(clamp(accumulated / weightSum, minimum4, maximum4), 1.);
}`;

// Lightweight browser port of Anime4K v3.2 Original's edge refinement.
// The official shader derives a Sobel direction and pulls edge pixels along
// that direction; this fused pass performs the same operation after scaling.
export const anime4kShader = `#version 300 es
precision highp float;
uniform sampler2D u_src;
uniform vec2 u_inSize;
in vec2 v_uv;
out vec4 outColor;
float luma(vec3 color) { return dot(color, vec3(.299,.587,.114)); }
float powerFunction(float x) {
  float x2=x*x, x3=x2*x, x4=x2*x2, x5=x2*x3;
  return 11.68129591*x5-42.46906057*x4+60.28286266*x3-41.84451327*x2+14.05517353*x-1.08152193;
}
void main() {
  vec2 d = 1. / u_inSize;
  float tl=luma(texture(u_src,v_uv+d*vec2(-1.,-1.)).rgb);
  float t =luma(texture(u_src,v_uv+d*vec2( 0.,-1.)).rgb);
  float tr=luma(texture(u_src,v_uv+d*vec2( 1.,-1.)).rgb);
  float l =luma(texture(u_src,v_uv+d*vec2(-1., 0.)).rgb);
  float r =luma(texture(u_src,v_uv+d*vec2( 1., 0.)).rgb);
  float bl=luma(texture(u_src,v_uv+d*vec2(-1., 1.)).rgb);
  float b =luma(texture(u_src,v_uv+d*vec2( 0., 1.)).rgb);
  float br=luma(texture(u_src,v_uv+d*vec2( 1., 1.)).rgb);
  vec2 gradient = vec2(-tl-2.*l-bl+tr+2.*r+br, -tl-2.*t-tr+bl+2.*b+br);
  float edge = clamp(length(gradient),0.,1.);
  float refine = clamp(powerFunction(edge) * .5,0.,1.);
  vec2 direction = gradient / max(length(gradient),.0001);
  vec2 pull = vec2(-sign(direction.x)*d.x, -sign(direction.y)*d.y);
  float ratio = abs(direction.x)/(abs(direction.x)+abs(direction.y)+.0001);
  vec3 xValue=texture(u_src,v_uv+vec2(pull.x,0.)).rgb;
  vec3 yValue=texture(u_src,v_uv+vec2(0.,pull.y)).rgb;
  vec3 base=texture(u_src,v_uv).rgb;
  outColor=vec4(clamp(mix(base,mix(yValue,xValue,ratio),refine),0.,1.),1.);
}`;

export const finalShader = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_src;
uniform vec2 u_size;
uniform float u_applyRcas;
out vec4 outColor;
vec3 fetchSource(ivec2 p) {
  return texelFetch(u_src, clamp(p,ivec2(0),ivec2(u_size)-1),0).rgb;
}
vec3 rcas(ivec2 p) {
  vec3 b=fetchSource(p+ivec2(0,-1));
  vec3 d=fetchSource(p+ivec2(-1,0));
  vec3 e=fetchSource(p);
  vec3 f=fetchSource(p+ivec2(1,0));
  vec3 h=fetchSource(p+ivec2(0,1));
  vec3 minimum4=min(min(b,d),min(f,h));
  vec3 maximum4=max(max(b,d),max(f,h));
  vec3 hitMin=minimum4/(4.*maximum4+1e-4);
  vec3 hitMax=(1.-maximum4)/(4.*minimum4-4.-1e-4);
  vec3 lobeRgb=max(-hitMin,hitMax);
  float lobe=max(-.1875,min(max(lobeRgb.r,max(lobeRgb.g,lobeRgb.b)),0.))*.87;
  return clamp((lobe*(b+d+f+h)+e)/(4.*lobe+1.),0.,1.);
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec3 plain=fetchSource(p);
  outColor=vec4(mix(plain,rcas(p),u_applyRcas),1.);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "Shader compilation failed");
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, fragmentSource: string, vertexSource = upscaleVertexShader) {
  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "Shader link failed");
  return program;
}

const INTERNAL_SIZE = 320;
const OUTPUT_SIZE = 640;

export class RelightRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private upscalePrograms: Record<SpatialUpscaleMethod, WebGLProgram>;
  private finalProgram: WebGLProgram;
  private videoTexture: WebGLTexture;
  private maskTexture: WebGLTexture;
  private relightTexture: WebGLTexture;
  private relightFramebuffer: WebGLFramebuffer;
  private upscaleTexture: WebGLTexture;
  private upscaleFramebuffer: WebGLFramebuffer;
  private neuralTexture: WebGLTexture;
  private neuralUpscaler: NeuralUpscaler;
  private neuralReadback = new Uint8Array(INTERNAL_SIZE * INTERNAL_SIZE * 4);
  private neuralReadbackCanvas = document.createElement("canvas");
  private neuralReadbackContext = this.neuralReadbackCanvas.getContext("2d", { alpha: false })!;
  private spatialFrameCount = 0;
  private spatialFpsStartedAt = performance.now();
  private measuredMethod?: UpscaleMethod;
  private maskWidth = 1;
  private maskHeight = 1;
  autoEv = 0;
  sourcePersonMean: [number, number, number] = [.5, .5, .5];
  sourcePersonStd: [number, number, number] = [.22, .22, .22];
  sourceBackgroundMean: [number, number, number] = [.5, .5, .5];
  sourceBackgroundStd: [number, number, number] = [.22, .22, .22];
  sourcePersonLabMean: [number, number, number] = [.5, 0, 0];
  sourcePersonLabStd: [number, number, number] = [.22, .22, .22];
  sourceBackgroundLabMean: [number, number, number] = [.5, 0, 0];
  sourceBackgroundLabStd: [number, number, number] = [.22, .22, .22];
  viewZoom = 1;
  viewCenter: [number, number] = [.5, .5];
  private framingInitialized = false;

  constructor(
    private canvas: HTMLCanvasElement,
    reportUpscaleStatus: (message: string, error?: boolean) => void = () => {},
    private reportFps: (fps: number, method: UpscaleMethod) => void = () => {}
  ) {
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
    if (!gl) throw new Error("This browser does not support WebGL 2");
    this.gl = gl;
    this.canvas.width = OUTPUT_SIZE;
    this.canvas.height = OUTPUT_SIZE;
    this.program = createProgram(gl, fragmentShader, vertexShader);
    this.upscalePrograms = {
      bilinear: createProgram(gl, bilinearShader),
      lanczos2: createProgram(gl, lanczos2Shader),
      fsr1: createProgram(gl, fsrEasuShader),
      anime4k: createProgram(gl, anime4kShader)
    };
    this.finalProgram = createProgram(gl, finalShader);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.videoTexture = this.createTexture(gl.LINEAR);
    this.maskTexture = this.createTexture(gl.LINEAR);
    this.neuralTexture = this.createTexture(gl.LINEAR);
    this.neuralUpscaler = new NeuralUpscaler(reportUpscaleStatus, this.reportFps);
    this.neuralReadbackCanvas.width = this.neuralReadbackCanvas.height = INTERNAL_SIZE;
    [this.relightTexture, this.relightFramebuffer] = this.createRenderTarget(INTERNAL_SIZE, gl.LINEAR);
    [this.upscaleTexture, this.upscaleFramebuffer] = this.createRenderTarget(OUTPUT_SIZE, gl.NEAREST);
    gl.useProgram(this.program);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_video"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_mask"), 1);
    Object.values(this.upscalePrograms).forEach((program) => {
      gl.useProgram(program);
      gl.uniform1i(gl.getUniformLocation(program, "u_src"), 0);
    });
    gl.useProgram(this.finalProgram);
    gl.uniform1i(gl.getUniformLocation(this.finalProgram, "u_src"), 0);
  }

  private createTexture(filter: number) {
    const { gl } = this;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));
    return texture;
  }

  private createRenderTarget(size: number, filter: number) {
    const { gl } = this;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const framebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("Upscaling framebuffer is incomplete");
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return [texture, framebuffer] as const;
  }

  updateMask(mask: Uint8Array, width: number, height: number) {
    const matte = new Uint8Array(mask);
    this.maskWidth = width;
    this.maskHeight = height;
    const { gl } = this;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, matte);
    return matte;
  }

  render(source: HTMLVideoElement | HTMLCanvasElement, look: Look, uploadSource = true) {
    const isVideo = source instanceof HTMLVideoElement;
    if (isVideo && source.readyState < 2) return;
    const width = isVideo ? source.videoWidth : source.width;
    const height = isVideo ? source.videoHeight : source.height;
    if (!width || !height) return;
    if (this.measuredMethod !== look.upscaleMethod) {
      this.measuredMethod = look.upscaleMethod;
      this.spatialFrameCount = 0;
      this.spatialFpsStartedAt = performance.now();
    }
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.relightFramebuffer);
    gl.viewport(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    if (uploadSource) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_autoEv"), look.autoAdjust && !look.labTransfer ? this.autoEv : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_manualEv"), look.autoAdjust ? look.manualEv : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_backgroundEv"), look.backgroundEv);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_warmth"), look.autoAdjust ? look.warmth : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_contrast"), look.autoAdjust ? look.contrast : 1);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_isolate"), look.isolate ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_matteThreshold"), look.matteThreshold);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_labTransfer"), look.labTransfer && look.autoAdjust ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_referenceStrength"), look.autoAdjust ? look.referenceStrength : 0);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourcePersonMean"), this.sourcePersonMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourcePersonStd"), this.sourcePersonStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourceBackgroundMean"), this.sourceBackgroundMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourceBackgroundStd"), this.sourceBackgroundStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referencePersonMean"), look.referencePersonMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referencePersonStd"), look.referencePersonStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundMean"), look.referenceBackgroundMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundStd"), look.referenceBackgroundStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourcePersonLabMean"), this.sourcePersonLabMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourcePersonLabStd"), this.sourcePersonLabStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourceBackgroundLabMean"), this.sourceBackgroundLabMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourceBackgroundLabStd"), this.sourceBackgroundLabStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referencePersonLabMean"), look.referencePersonLabMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referencePersonLabStd"), look.referencePersonLabStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundLabMean"), look.referenceBackgroundLabMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundLabStd"), look.referenceBackgroundLabStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundColor"), look.referenceBackgroundColor);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_replaceBackground"), look.replaceBackground && look.referenceStrength > 0 ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_maskTexel"), 1 / this.maskWidth, 1 / this.maskHeight);
    const audioGain = look.audioReactive ? .6 + look.audioLevel * .75 + look.audioBeat * .85 : 1;
    gl.uniform1f(gl.getUniformLocation(this.program, "u_rimIntensity"), Math.min(1, look.rimIntensity * audioGain));
    gl.uniform1f(gl.getUniformLocation(this.program, "u_showLight"), look.showLight ? 1 : 0);
    gl.uniform2fv(gl.getUniformLocation(this.program, "u_lightPosition"), look.lightPosition);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_rimColor"), look.rimColor);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_viewZoom"), this.viewZoom);
    gl.uniform2fv(gl.getUniformLocation(this.program, "u_viewCenter"), this.viewCenter);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    const neuralMethod = look.upscaleMethod === "websr" || look.upscaleMethod === "fsrcnn" ? look.upscaleMethod : undefined;
    const spatialMethod: SpatialUpscaleMethod = neuralMethod ? "bilinear" : look.upscaleMethod as SpatialUpscaleMethod;
    const upscaleProgram = this.upscalePrograms[spatialMethod];
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.upscaleFramebuffer);
    gl.viewport(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
    gl.useProgram(upscaleProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.relightTexture);
    gl.uniform2f(gl.getUniformLocation(upscaleProgram, "u_inSize"), INTERNAL_SIZE, INTERNAL_SIZE);
    gl.uniform2f(gl.getUniformLocation(upscaleProgram, "u_outSize"), OUTPUT_SIZE, OUTPUT_SIZE);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
    gl.useProgram(this.finalProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.upscaleTexture);
    gl.uniform2f(gl.getUniformLocation(this.finalProgram, "u_size"), OUTPUT_SIZE, OUTPUT_SIZE);
    gl.uniform1f(gl.getUniformLocation(this.finalProgram, "u_applyRcas"), look.rcasEnabled ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (neuralMethod) {
      // Read the shared 320px working frame only when the async upscaler is
      // ready for another request. This avoids both a per-frame GPU readback
      // and an accidental baseline-upscale pass before neural inference.
      this.neuralUpscaler.request(neuralMethod, () => this.captureNeuralInput());
      const neuralResult = this.neuralUpscaler.getResult(neuralMethod);
      if (neuralResult) this.drawNeuralResult(neuralResult, look.rcasEnabled);
    } else {
      this.measureSpatialFps(spatialMethod);
    }
  }

  private measureSpatialFps(method: SpatialUpscaleMethod) {
    this.spatialFrameCount++;
    const now = performance.now();
    const elapsed = now - this.spatialFpsStartedAt;
    if (elapsed < 750) return;
    this.reportFps(this.spatialFrameCount * 1000 / elapsed, method);
    this.spatialFrameCount = 0;
    this.spatialFpsStartedAt = now;
  }

  private captureNeuralInput() {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.relightFramebuffer);
    gl.readPixels(0, 0, INTERNAL_SIZE, INTERNAL_SIZE, gl.RGBA, gl.UNSIGNED_BYTE, this.neuralReadback);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const pixels = new Uint8ClampedArray(
      this.neuralReadback.buffer,
      this.neuralReadback.byteOffset,
      this.neuralReadback.byteLength
    );
    this.neuralReadbackContext.putImageData(new ImageData(pixels, INTERNAL_SIZE, INTERNAL_SIZE), 0, 0);
    return this.neuralReadbackCanvas;
  }

  private drawNeuralResult(result: HTMLCanvasElement, applyRcas: boolean) {
    const { gl } = this;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.neuralTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, result);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
    gl.useProgram(this.finalProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.neuralTexture);
    gl.uniform2f(gl.getUniformLocation(this.finalProgram, "u_size"), result.width, result.height);
    gl.uniform1f(gl.getUniformLocation(this.finalProgram, "u_applyRcas"), applyRcas ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  updateFraming(mask: Uint8Array, width: number, height: number, threshold = .44, maxZoom = 4) {
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let count = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y * width + x] < Math.round(Math.min(.95, threshold) * 255)) continue;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        count++;
      }
    }
    if (count < 80 || maxY <= minY) return;
    const personHeight = (maxY - minY + 1) / height;
    const targetZoom = Math.max(.65, Math.min(maxZoom, .8 / personHeight));
    const targetCenter: [number, number] = [
      (minX + maxX + 1) / (2 * width),
      (minY + maxY + 1) / (2 * height)
    ];
    if (!this.framingInitialized) {
      this.viewZoom = targetZoom;
      this.viewCenter = targetCenter;
      this.framingInitialized = true;
      return;
    }
    this.viewZoom += (targetZoom - this.viewZoom) * .08;
    this.viewCenter[0] += (targetCenter[0] - this.viewCenter[0]) * .14;
    this.viewCenter[1] += (targetCenter[1] - this.viewCenter[1]) * .14;
  }

  resetFraming() {
    this.viewZoom = 1;
    this.viewCenter = [.5, .5];
    this.framingInitialized = false;
  }

  get maskSize() { return [this.maskWidth, this.maskHeight] as const; }
}

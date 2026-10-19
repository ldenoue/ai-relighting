export type Look = {
  autoAdjust: boolean;
  targetLuma: number;
  manualEv: number;
  backgroundEv: number;
  warmth: number;
  contrast: number;
  isolate: boolean;
  referenceStrength: number;
  referencePersonMean: [number, number, number];
  referencePersonStd: [number, number, number];
  referenceBackgroundMean: [number, number, number];
  referenceBackgroundStd: [number, number, number];
  referenceBackgroundColor: [number, number, number];
  replaceBackground: boolean;
  rimIntensity: number;
  rimColor: [number, number, number];
};

const vertexShader = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_position.x * .5 + .5, 1. - (a_position.y * .5 + .5));
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
uniform float u_referenceStrength;
uniform vec3 u_sourcePersonMean;
uniform vec3 u_sourcePersonStd;
uniform vec3 u_sourceBackgroundMean;
uniform vec3 u_sourceBackgroundStd;
uniform vec3 u_referencePersonMean;
uniform vec3 u_referencePersonStd;
uniform vec3 u_referenceBackgroundMean;
uniform vec3 u_referenceBackgroundStd;
uniform vec3 u_referenceBackgroundColor;
uniform float u_replaceBackground;
uniform vec2 u_maskTexel;
uniform float u_rimIntensity;
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

vec3 screenLight(vec3 base, vec3 light) {
  return 1. - (1. - base) * (1. - clamp(light, 0., 1.));
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
  float person = mix(1., smoothstep(.12, .88, matteAlpha), u_isolate);
  vec3 foreground = transferLook(source, u_sourcePersonMean, u_sourcePersonStd, u_referencePersonMean, u_referencePersonStd, 1.);
  vec3 background = transferLook(source, u_sourceBackgroundMean, u_sourceBackgroundStd, u_referenceBackgroundMean, u_referenceBackgroundStd, 0.);
  foreground = grade(foreground, u_autoEv + u_manualEv);

  // Approximate a local signed-distance field around the matte boundary. The
  // near ring yields a one-to-two-pixel rim; a farther ring identifies only
  // pixels a few samples inside the silhouette for restrained light spill.
  vec2 edgeRadius = d * 1.5;
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
  float innerBand = smoothstep(.1, .5, max(contourGradient, confidenceEdge * .72))
    * smoothstep(.28, .72, matteAlpha);

  vec2 spillRadius = d * 4.5;
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
  vec2 lightCenter = vec2(.5, .35);
  vec2 wideDelta = (v_uv - lightCenter) / vec2(.48, .42);
  vec2 coreDelta = (v_uv - lightCenter) / vec2(.22, .2);
  float wideHalo = exp(-dot(wideDelta, wideDelta) * 1.35);
  float mediumHalo = exp(-dot(wideDelta, wideDelta) * 4.8);
  float hotCore = exp(-dot(coreDelta, coreDelta) * 3.2);
  float irradiance = clamp(wideHalo * .12 + mediumHalo * .28 + hotCore * .72, 0., 1.);

  vec2 floorDelta = (v_uv - vec2(.5, .92)) / vec2(.3, .055);
  float floorPool = exp(-dot(floorDelta, floorDelta) * 2.2) * .2;

  vec3 rimColor = u_rimColor;
  float sourceLuma = dot(foreground, vec3(.2126, .7152, .0722));
  float edgeTransmission = mix(1.28, .72, matteAlpha);
  float materialResponse = mix(1.08, .62, sourceLuma);
  float rim = innerBand * mix(.55, 1., irradiance) * edgeTransmission * materialResponse
    * u_rimIntensity * 1.2 * u_isolate;
  float spill = inwardBand * mix(.4, 1., irradiance) * materialResponse
    * u_rimIntensity * .13 * u_isolate;
  foreground = screenLight(foreground, rimColor * (rim + spill));

  background = grade(background, u_backgroundEv);
  vec3 solidBackground = clamp(u_referenceBackgroundColor * exp2(u_backgroundEv), 0., 1.);
  background = mix(background, solidBackground, u_replaceBackground);
  vec3 behindLight = rimColor * (irradiance + floorPool) * u_rimIntensity * u_isolate;
  background = screenLight(background, behindLight);

  // Compositing with the continuous alpha performs the occlusion: solid body
  // pixels hide the source, while fractional edge/hair pixels reveal a
  // proportional amount of the halo and receive stronger scattering.
  vec3 composite = mix(background, foreground, person);
  outColor = vec4(clamp(composite, 0., 1.), 1.);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "Shader compilation failed");
  return shader;
}

export class RelightRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private videoTexture: WebGLTexture;
  private maskTexture: WebGLTexture;
  private maskWidth = 1;
  private maskHeight = 1;
  autoEv = 0;
  sourcePersonMean: [number, number, number] = [.5, .5, .5];
  sourcePersonStd: [number, number, number] = [.22, .22, .22];
  sourceBackgroundMean: [number, number, number] = [.5, .5, .5];
  sourceBackgroundStd: [number, number, number] = [.22, .22, .22];
  viewZoom = 1;
  viewCenter: [number, number] = [.5, .5];
  private framingInitialized = false;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
    if (!gl) throw new Error("This browser does not support WebGL 2");
    this.gl = gl;
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, compile(gl, gl.VERTEX_SHADER, vertexShader));
    gl.attachShader(this.program, compile(gl, gl.FRAGMENT_SHADER, fragmentShader));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program) ?? "Shader link failed");

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    this.videoTexture = this.createTexture(gl.LINEAR);
    this.maskTexture = this.createTexture(gl.LINEAR);
    gl.useProgram(this.program);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_video"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_mask"), 1);
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

  render(source: HTMLVideoElement | HTMLCanvasElement, look: Look) {
    const isVideo = source instanceof HTMLVideoElement;
    if (isVideo && source.readyState < 2) return;
    const width = isVideo ? source.videoWidth : source.width;
    const height = isVideo ? source.videoHeight : source.height;
    if (!width || !height) return;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const { gl } = this;
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_autoEv"), look.autoAdjust ? this.autoEv : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_manualEv"), look.autoAdjust ? look.manualEv : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_backgroundEv"), look.backgroundEv);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_warmth"), look.autoAdjust ? look.warmth : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_contrast"), look.autoAdjust ? look.contrast : 1);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_isolate"), look.isolate ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_referenceStrength"), look.autoAdjust ? look.referenceStrength : 0);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourcePersonMean"), this.sourcePersonMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourcePersonStd"), this.sourcePersonStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourceBackgroundMean"), this.sourceBackgroundMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_sourceBackgroundStd"), this.sourceBackgroundStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referencePersonMean"), look.referencePersonMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referencePersonStd"), look.referencePersonStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundMean"), look.referenceBackgroundMean);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundStd"), look.referenceBackgroundStd);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_referenceBackgroundColor"), look.referenceBackgroundColor);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_replaceBackground"), look.replaceBackground && look.referenceStrength > 0 ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_maskTexel"), 1 / this.maskWidth, 1 / this.maskHeight);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_rimIntensity"), look.rimIntensity);
    gl.uniform3fv(gl.getUniformLocation(this.program, "u_rimColor"), look.rimColor);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_viewZoom"), this.viewZoom);
    gl.uniform2fv(gl.getUniformLocation(this.program, "u_viewCenter"), this.viewCenter);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  updateFraming(mask: Uint8Array, width: number, height: number) {
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let count = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y * width + x] < 112) continue;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        count++;
      }
    }
    if (count < 80 || maxY <= minY) return;
    const personHeight = (maxY - minY + 1) / height;
    const targetZoom = Math.max(.65, Math.min(4, .8 / personHeight));
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

  get maskSize() { return [this.maskWidth, this.maskHeight] as const; }
}

import {
  anime4kShader,
  bilinearShader,
  finalShader,
  fsrEasuShader,
  lanczos2Shader,
  upscaleVertexShader,
  type UpscaleMethod
} from "./renderer";

export type SpatialImageMethod = Exclude<UpscaleMethod, "websr" | "fsrcnn">;

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "Shader compilation failed");
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, fragmentSource: string) {
  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, upscaleVertexShader));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "Shader link failed");
  }
  return program;
}

export class ImageSpatialUpscaler {
  private gl: WebGL2RenderingContext;
  private programs: Record<SpatialImageMethod, WebGLProgram>;
  private finalProgram: WebGLProgram;
  private inputTexture: WebGLTexture;
  private intermediateTexture?: WebGLTexture;
  private intermediateFramebuffer?: WebGLFramebuffer;
  private targetWidth = 0;
  private targetHeight = 0;

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: true
    });
    if (!gl) throw new Error("WebGL 2 is unavailable");
    this.gl = gl;
    this.programs = {
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

    this.inputTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.inputTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    Object.values(this.programs).forEach((program) => {
      gl.useProgram(program);
      gl.uniform1i(gl.getUniformLocation(program, "u_src"), 0);
    });
    gl.useProgram(this.finalProgram);
    gl.uniform1i(gl.getUniformLocation(this.finalProgram, "u_src"), 0);
  }

  private ensureTarget(width: number, height: number) {
    if (width === this.targetWidth && height === this.targetHeight) return;
    const { gl } = this;
    const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (width > maxSize || height > maxSize) {
      throw new Error(`Output exceeds this GPU's ${maxSize}px texture limit`);
    }
    if (this.intermediateTexture) gl.deleteTexture(this.intermediateTexture);
    if (this.intermediateFramebuffer) gl.deleteFramebuffer(this.intermediateFramebuffer);
    this.canvas.width = width;
    this.canvas.height = height;
    this.targetWidth = width;
    this.targetHeight = height;

    this.intermediateTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.intermediateTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    this.intermediateFramebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.intermediateFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.intermediateTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("Could not create the upscaling framebuffer");
    }
  }

  run(
    method: SpatialImageMethod,
    source: TexImageSource,
    inputWidth: number,
    inputHeight: number,
    outputWidth: number,
    outputHeight: number,
    applyRcas: boolean
  ) {
    const startedAt = performance.now();
    const { gl } = this;
    this.ensureTarget(outputWidth, outputHeight);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.inputTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

    const program = this.programs[method];
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.intermediateFramebuffer!);
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.uniform2f(gl.getUniformLocation(program, "u_inSize"), inputWidth, inputHeight);
    gl.uniform2f(gl.getUniformLocation(program, "u_outSize"), outputWidth, outputHeight);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(this.finalProgram);
    gl.bindTexture(gl.TEXTURE_2D, this.intermediateTexture!);
    gl.uniform2f(gl.getUniformLocation(this.finalProgram, "u_size"), outputWidth, outputHeight);
    gl.uniform1f(gl.getUniformLocation(this.finalProgram, "u_applyRcas"), applyRcas ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.finish();
    return performance.now() - startedAt;
  }
}

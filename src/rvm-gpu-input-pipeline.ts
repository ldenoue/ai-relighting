import { Tensor, getWebGpuDevice } from "@litertjs/core";
import tgpu, { d, std, type TgpuRoot } from "typegpu";

/**
 * Converts a synchronized square source frame into RVM's normalized NHWC RGB
 * tensor without reading pixels back to JavaScript. TypeGPU and LiteRT share
 * the GPUDevice, and the same storage buffer is reused for every inference.
 */
export class RvmGpuInputPipeline {
  readonly #root: TgpuRoot;
  readonly #sourceTexture;
  readonly #inputBuffer: GPUBuffer;
  readonly #preprocess;

  private constructor(
    readonly sourceWidth: number,
    readonly sourceHeight: number,
    readonly modelWidth: number,
    readonly modelHeight: number,
    device: GPUDevice
  ) {
    this.#root = tgpu.initFromDevice({ device });
    this.#sourceTexture = this.#root
      .createTexture({
        size: [sourceWidth, sourceHeight],
        format: "rgba8unorm"
      })
      .$usage("sampled", "render");
    const sourceView = this.#sourceTexture.createView();
    const sampler = this.#root.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });

    const matteSize = Math.min(modelWidth, modelHeight);
    const squareOffsetX = (modelWidth - matteSize) / 2;
    this.#inputBuffer = device.createBuffer({
      label: `RVM ${modelWidth}x${modelHeight} normalized RGB input`,
      size: modelWidth * modelHeight * 3 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    const input = this.#root.createMutable(
      d.arrayOf(d.f32, modelWidth * modelHeight * 3),
      this.#inputBuffer
    );

    this.#preprocess = this.#root.createGuardedComputePipeline((x, y) => {
      "use gpu";
      const offset = (y * modelWidth + x) * 3;
      if (x < squareOffsetX || x >= squareOffsetX + matteSize) {
        input.$[offset] = 0;
        input.$[offset + 1] = 0;
        input.$[offset + 2] = 0;
      } else {
        const uv = d.vec2f(
          (d.f32(x) - squareOffsetX + 0.5) / matteSize,
          (d.f32(y) + 0.5) / matteSize
        );
        const color = std.textureSampleLevel(sourceView.$, sampler.$, uv, 0);
        input.$[offset] = color.r;
        input.$[offset + 1] = color.g;
        input.$[offset + 2] = color.b;
      }
    });
  }

  static create(sourceWidth: number, sourceHeight: number, modelWidth: number, modelHeight: number) {
    const device = getWebGpuDevice();
    if (!device) throw new Error("LiteRT did not expose its WebGPU device");
    return new RvmGpuInputPipeline(sourceWidth, sourceHeight, modelWidth, modelHeight, device);
  }

  matches(sourceWidth: number, sourceHeight: number) {
    return this.sourceWidth === sourceWidth && this.sourceHeight === sourceHeight;
  }

  createInputTensor(source: GPUCopyExternalImageSource) {
    this.#sourceTexture.write(source);
    this.#preprocess.dispatchThreads(this.modelWidth, this.modelHeight);
    return new Tensor(
      this.#inputBuffer,
      [1, this.modelHeight, this.modelWidth, 3],
      "float32"
    );
  }

  destroy() {
    this.#sourceTexture.destroy();
    this.#inputBuffer.destroy();
    this.#root.destroy();
  }
}

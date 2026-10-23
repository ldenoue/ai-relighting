# Third-party notices

## AMD FidelityFX Super Resolution 1.0

The WebGL2 EASU and RCAS shaders are adapted from AMD FidelityFX Super Resolution 1.0 and the non-gather WebGL2 port in [natureglass/Brewser-apps FSR Lab](https://github.com/natureglass/Brewser-apps/blob/main/apps/com.natureglass.fsrlab/index.html).

Copyright © 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Anime4K

The edge-directed upscaler is a fused WebGL2 adaptation of [Anime4K v3.2 Upscale Original](https://github.com/bloc97/Anime4K/blob/master/glsl/Upscale/Anime4K_Upscale_Original_x2.glsl).

Copyright © 2019 bloc97

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## WebSR

The WebGPU CNN runner and bundled `cnn-2x-s-rl` real-life weights come from [sb2702/websr](https://github.com/sb2702/websr).

Copyright © 2023 Sam Bhattacharyya

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## FSRCNN

`public/models/fsrcnn-x2-320.onnx` is a browser-targeted ONNX conversion of the ×2 TensorFlow model from [Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow). It was converted from the original fixed graph to a fixed 320 × 320 luminance input and 640 × 640 output; the weights were not retrained. The source project is distributed under Apache License 2.0. A copy is included at [licenses/FSRCNN-APACHE-2.0.txt](licenses/FSRCNN-APACHE-2.0.txt).

FSRCNN inference uses [Microsoft ONNX Runtime Web](https://github.com/microsoft/onnxruntime), distributed under the MIT License.

## Robust Video Matting

`public/models/rvm-mobilenetv3-fp32.onnx` is the official MobileNetV3 FP32 ONNX export from [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting). It is used at 256 × 256 at full internal resolution through Microsoft ONNX Runtime Web's WASM execution provider.

`public/models/rvm-tfjs/model.json` and `public/models/rvm-tfjs/group1-shard1of1.bin` are the official INT8 TensorFlow.js export from the same project’s `tfjs` branch. They provide the WebGPU-accelerated browser path through TensorFlow.js. TensorFlow.js is distributed under Apache License 2.0; the RVM graph and weights remain covered by RVM's GPL-3.0 license below.

Robust Video Matting was developed at ByteDance Inc. by Shanchuan Lin, Linjie Yang, Imran Saleemi, and Soumyadip Sengupta. The upstream project was re-released under the GNU General Public License v3.0. A copy is included at [licenses/RVM-GPL-3.0.txt](licenses/RVM-GPL-3.0.txt). Distribution of this model therefore introduces GPL-3.0 obligations; review compatibility with the intended distribution model before treating this prototype as a proprietary product.

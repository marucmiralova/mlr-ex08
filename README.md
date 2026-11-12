# MLR exercise 08 — live leaf classifier in the browser

**Live page: <https://marucmiralova.github.io/mlr-ex08/>**

Open that address on a phone (or scan the QR code shown at the bottom of the page),
tap **Start rear camera**, and the page classifies whatever is in the central square of
the camera image, continuously. A **Choose photo…** fallback works when the camera is
blocked or unavailable.

Everything runs locally in the browser: [`leaf_model.onnx`](leaf_model.onnx) is executed by
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) loaded from a CDN
(jsDelivr). Nothing is uploaded anywhere — camera frames never leave the device.

## Preprocessing (exactly as described in [`model_info.json`](model_info.json))

| step | detail |
| --- | --- |
| crop | central square of the frame (`min(w,h)`, centred) |
| resize | to `input_size` × `input_size` = 96 × 96 |
| scale | RGB samples to 0…1 (divide by 255) |
| standardise | per channel `(x − mean) / std`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| tensor | `1 × 3 × 96 × 96` float32, NCHW, fed to input `image` |

The model output `logits` is turned into probabilities with a softmax, and the probability
of every class ([`model_info.json`](model_info.json): `apple`, `cherry`, `chestnut`,
`maple`) is shown as a bar, highest first, updated on every processed frame.

## Self-check

The **Self-test (check_image.jpg)** button runs [`check_image.jpg`](check_image.jpg)
through exactly the same pipeline as a camera frame and compares the result with
[`check_expected.json`](check_expected.json) in a table, ending with PASS/FAIL.
Measured in headless Chromium the largest per-class difference is **1.3 × 10⁻⁶**
(apple 0.983354 vs 0.983353 expected), so the page agrees with the reference values.

## Files

| file | purpose |
| --- | --- |
| `index.html` | the whole application (markup, styles, JavaScript — no build step) |
| `leaf_model.onnx` | the trained leaf classifier |
| `model_info.json` | classes, input size, channel mean/std |
| `check_image.jpg`, `check_expected.json` | self-test image and its expected probabilities |
| `.nojekyll` | serve the page as-is, without Jekyll processing |

## Notes on the implementation

* The camera is requested with `facingMode: {ideal: "environment"}` so the rear camera is
  used, with graceful fallbacks if that is refused.
* For iOS Safari: the video element is `playsinline muted`, frames are read with
  `drawImage` (not `createImageBitmap`), and `ort.env.wasm.numThreads = 1` because
  GitHub Pages does not send the cross-origin-isolation headers that `SharedArrayBuffer`
  would need. A second pinned onnxruntime-web build is tried automatically if the first
  one cannot initialise.
* The QR code at the bottom is generated **by an encoder written inside `index.html`**
  (byte mode, EC level M, ISO/IEC 18004) — no third-party QR image service is involved.
  It encodes the deployed GitHub Pages address, and the same address is next to it as a
  clickable link.
* Camera access requires a secure context, i.e. HTTPS (or `localhost`).

## Run it locally

Any static server works, e.g. `python -m http.server 8000` and then open
`http://localhost:8000/`. `localhost` counts as a secure context, so the camera works
there as well.

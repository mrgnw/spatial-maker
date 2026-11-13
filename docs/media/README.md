# Demo media

`coffee.png` is a 600 × 400 photograph by Rachel Michetti, courtesy of Pikolo Espresso Bar, released under CC0. Source: [scikit-image v0.20.0](https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/data/coffee.png); [license and credit](https://scikit-image.org/docs/0.20.x/api/skimage.data.html#skimage.data.coffee). It contains no people. ExifTool reported no GPS metadata in the input or committed outputs.

`coffee-sbs.png` and `coffee-spatial.heic` are real outputs from the v0.8.0 macOS release binary using Depth Anything V2 Small (`s`, Apache-2.0). No weights are stored here. Output quality and playback on a Vision Pro have not been checked on a device.

## Reproduce the conversion

Run from the repository root with spatial-maker installed:

```sh
INPUT="$PWD/docs/media/coffee.png"
OUTPUT=/tmp/spatial-demo/coffee
```

```sh
mkdir -p /tmp/spatial-demo
spatial-maker "$INPUT" -o "$OUTPUT" -m s --output-types spatial,sbs --force
```

This writes `/tmp/spatial-demo/coffee-spatial.png` and `/tmp/spatial-demo/coffee-spatial.heic`. The committed copies are renamed for readability.

## Terminal recording

`conversion.cast` is an unedited asciinema recording of the release binary, captured on an Apple Silicon Mac in `/tmp/spatial-demo` with the prompt set to `$ `. The model had already downloaded. The recorded conversion uses `--force` to regenerate depth; it does not reuse a saved depth map.

```sh
spatial-maker --version
```

```sh
spatial-maker coffee.png -m s --output-types spatial,sbs --force
```

`conversion.gif` was rendered with agg 1.9.0 at 90 columns, 14 rows, 14 px type, GitHub dark theme, a 10 fps cap, and idle gaps limited to one second. Processing frames are real; playback compresses idle time.

## Preview images

`hero.png` places the original photo beside a red–cyan anaglyph made from the actual side-by-side output. Red goes to the left eye, cyan to the right. The anaglyph is a documentation preview, not an output format of spatial-maker. It was made with ffmpeg:

```sh
INPUT="$PWD/docs/media/coffee-sbs.png"
OUTPUT=/tmp/spatial-demo/coffee-anaglyph.png
```

```sh
ffmpeg -i "$INPUT" -vf stereo3d=sbsl:arcc -frames:v 1 -y "$OUTPUT"
```

`social-preview.png` is a 1280 × 640 composition of the same original and anaglyph, with project title and format labels. The owner can upload it under repository **Settings → General → Social preview**. These preview layouts were rasterized from SVG with resvg; no generated or retouched stereo result was substituted.

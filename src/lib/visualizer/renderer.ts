// WebGL compositor for the wall visualizer.
// Every material change is one draw call in the browser. No server call.
import type { Plane } from "./geometry";

export const MAX_PLANES = 24;

export type Material = {
  slug: string;
  name: string;
  finish: string;
  /** Sheet image (full panel, portrait). Empty for plain colours. */
  texture: string;
  color: [number, number, number];
  /** Panel size in metres (width, height). */
  tile: [number, number];
  /** 0..1 looks. */
  gloss: number;
  metal: number;
  sparkle: number;
  mirror: number;
  /** 0 = flat, 1 = fluted 3D panel. */
  relief: number;
};

export type View = {
  material: Material;
  /** Size multiplier chosen by the user. */
  scale: number;
  joints: boolean;
  /** Turn the sheet by 90 degrees. */
  turned: boolean;
  /** Before / after split, 0..1 from the left. Pixels to the right of the split show the product. */
  split: number;
  /** Show a tint on the selected area instead of the product. */
  highlight: number;
};

const VERTEX = `#version 300 es
in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = vec2(aPosition.x * 0.5 + 0.5, 0.5 - aPosition.y * 0.5);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uPhoto;
uniform sampler2D uMask;
uniform sampler2D uLabels;
uniform sampler2D uShade;
uniform sampler2D uAlbedo;
uniform vec3 uN[${MAX_PLANES}];
uniform vec3 uD[${MAX_PLANES}];
uniform vec3 uUp[${MAX_PLANES}];
uniform float uH[${MAX_PLANES}];
uniform float uRef[${MAX_PLANES}];
uniform vec2 uExtent;      // image size divided by its longer side
uniform float uFocal;
uniform float uScale;
uniform vec2 uTile;
uniform vec3 uColor;
uniform float uUseTexture;
uniform float uJoints;
uniform float uTurned;
uniform float uGloss;
uniform float uMetal;
uniform float uSparkle;
uniform float uMirror;
uniform float uRelief;
uniform float uSplit;
uniform float uHighlight;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  vec3 photo = texture(uPhoto, vUv).rgb;
  float alpha = texture(uMask, vUv).r;
  int id = int(texture(uLabels, vUv).r * 255.0 + 0.5);
  if (alpha < 0.004 || id >= ${MAX_PLANES} || vUv.x < uSplit) {
    outColor = vec4(photo, 1.0);
    return;
  }
  vec3 n = uN[id];
  vec3 ray = vec3((vUv - 0.5) * uExtent / uFocal, 1.0);
  float t = uH[id] / dot(n, ray);
  vec3 point = ray * t;
  vec2 metres = vec2(dot(point, uD[id]), -dot(point, uUp[id])) / uScale;
  if (uTurned > 0.5) metres = vec2(metres.y, -metres.x);

  vec2 tile = metres / uTile;
  vec2 cell = floor(tile);
  vec2 inside = fract(tile);
  // Mirror every second sheet so the pattern does not repeat in an obvious way.
  vec2 sheet = vec2(mod(cell.x, 2.0) > 0.5 ? 1.0 - inside.x : inside.x, mod(cell.y, 2.0) > 0.5 ? 1.0 - inside.y : inside.y);
  vec2 dx = dFdx(tile);
  vec2 dy = dFdy(tile);
  vec3 albedo = mix(uColor, textureGrad(uAlbedo, sheet, dx, dy).rgb, uUseTexture);

  // Fluted 3D panel: ribs 40 mm wide that catch the light.
  float relief = 1.0;
  if (uRelief > 0.5) {
    float rib = fract(metres.x / 0.04);
    float pixel = fwidth(metres.x / 0.04);
    float profile = sin(rib * 3.14159);
    float slope = cos(rib * 3.14159);
    float detail = 1.0 - smoothstep(0.35, 1.2, pixel);
    relief = 1.0 + detail * (0.16 * slope - 0.34 * pow(1.0 - profile, 3.0));
  }

  // Panel joints: a dark 6 mm groove with a soft edge.
  vec2 edge = min(inside, 1.0 - inside) * uTile;
  float nearest = min(edge.x, edge.y);
  float blur = max(fwidth(nearest), 0.0005);
  float groove = (1.0 - smoothstep(0.003, 0.003 + blur, nearest)) * uJoints * clamp(0.02 / blur, 0.0, 1.0);

  // Light of the original wall.
  float shade = texture(uShade, vUv).r / max(uRef[id], 0.04);
  shade = clamp(shade, 0.22, 1.7);
  vec3 lit = albedo * relief * shade;

  // Surface looks.
  float facing = abs(dot(normalize(ray), n));
  float grazing = pow(1.0 - facing, 2.0);
  lit += uGloss * (0.10 * max(shade - 0.9, 0.0) + 0.06 * grazing);
  float brushed = hash(vec2(floor(metres.y * 900.0), 3.0)) - 0.5;
  lit *= 1.0 + uMetal * (0.10 * brushed + 0.22 * grazing - 0.05);
  vec2 grain = floor(metres * 1400.0);
  float fleck = step(0.985, hash(grain)) * hash(grain + 7.0);
  lit += uSparkle * fleck * 0.55 * shade * clamp(0.004 / max(fwidth(metres.x), 0.0001), 0.0, 1.0);
  vec3 sky = mix(vec3(0.92, 0.90, 0.86), vec3(0.62, 0.74, 0.90), clamp(0.5 - (vUv.y - 0.5) * 1.6, 0.0, 1.0));
  lit = mix(lit, albedo * (0.35 + 0.75 * sky) * (0.6 + 0.4 * shade), uMirror);

  lit = mix(lit, lit * 0.32, groove);
  lit = mix(lit, mix(photo, vec3(0.93, 0.22, 0.18), 0.45), uHighlight);
  outColor = vec4(mix(photo, clamp(lit, 0.0, 1.0), alpha), 1.0);
}`;

export class Renderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private textures: Record<string, WebGLTexture> = {};
  private uniforms = new Map<string, WebGLUniformLocation | null>();
  private planes: Plane[] = [];
  private refs: number[] = [];
  private focal = 0.8;
  private extent: [number, number] = [1, 1];
  private albedoFor = "";
  private pendingAlbedo = "";
  onTextureLoaded: (() => void) | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL 2 is not available in this browser.");
    this.gl = gl;
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || "Shader failed to compile.");
      return shader;
    };
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || "Shader program failed to link.");
    this.program = program;
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    ["uPhoto", "uMask", "uLabels", "uShade", "uAlbedo"].forEach((name, unit) => {
      const texture = gl.createTexture()!;
      this.textures[name] = texture;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
      const nearest = name === "uLabels";
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, nearest ? gl.NEAREST : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, nearest ? gl.NEAREST : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.uniform1i(this.location(name), unit);
    });
  }

  private location(name: string) {
    if (!this.uniforms.has(name)) this.uniforms.set(name, this.gl.getUniformLocation(this.program, name));
    return this.uniforms.get(name)!;
  }
  private bind(name: string) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + ["uPhoto", "uMask", "uLabels", "uShade", "uAlbedo"].indexOf(name));
    gl.bindTexture(gl.TEXTURE_2D, this.textures[name]);
  }

  setPhoto(source: ImageBitmap | HTMLCanvasElement, focal: number) {
    const gl = this.gl;
    this.canvas.width = source.width;
    this.canvas.height = source.height;
    gl.viewport(0, 0, source.width, source.height);
    this.bind("uPhoto");
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    const longer = Math.max(source.width, source.height);
    this.extent = [source.width / longer, source.height / longer];
    this.focal = focal;
  }
  private bytes(name: string, data: Uint8Array, width: number, height: number) {
    const gl = this.gl;
    this.bind(name);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  }
  setShade(data: Uint8Array, width: number, height: number) {
    this.bytes("uShade", data, width, height);
  }
  setSelection(mask: Uint8Array, labels: Uint8Array, width: number, height: number, planes: Plane[], refs: number[]) {
    this.bytes("uMask", mask, width, height);
    this.bytes("uLabels", labels, width, height);
    this.planes = planes;
    this.refs = refs;
  }
  setPlanes(planes: Plane[]) {
    this.planes = planes;
  }

  private loadAlbedo(url: string) {
    if (this.albedoFor === url || this.pendingAlbedo === url) return;
    this.pendingAlbedo = url;
    const image = new Image();
    image.onload = () => {
      if (this.pendingAlbedo !== url) return;
      const gl = this.gl;
      this.bind("uAlbedo");
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      this.albedoFor = url;
      this.pendingAlbedo = "";
      this.onTextureLoaded?.();
    };
    image.src = url;
  }

  draw(view: View) {
    const gl = this.gl;
    const m = view.material;
    if (m.texture) this.loadAlbedo(m.texture);
    const textured = m.texture && this.albedoFor === m.texture ? 1 : 0;
    const count = Math.min(MAX_PLANES, this.planes.length);
    const n = new Float32Array(MAX_PLANES * 3);
    const d = new Float32Array(MAX_PLANES * 3);
    const up = new Float32Array(MAX_PLANES * 3);
    const h = new Float32Array(MAX_PLANES).fill(1);
    const ref = new Float32Array(MAX_PLANES).fill(0.6);
    for (let i = 0; i < count; i++) {
      n.set(this.planes[i].n, i * 3);
      d.set(this.planes[i].d, i * 3);
      up.set(this.planes[i].up, i * 3);
      h[i] = this.planes[i].h;
      ref[i] = this.refs[i] ?? 0.6;
    }
    gl.uniform3fv(this.location("uN"), n);
    gl.uniform3fv(this.location("uD"), d);
    gl.uniform3fv(this.location("uUp"), up);
    gl.uniform1fv(this.location("uH"), h);
    gl.uniform1fv(this.location("uRef"), ref);
    gl.uniform2f(this.location("uExtent"), this.extent[0], this.extent[1]);
    gl.uniform1f(this.location("uFocal"), this.focal);
    gl.uniform1f(this.location("uScale"), view.scale);
    gl.uniform2f(this.location("uTile"), m.tile[0], m.tile[1]);
    gl.uniform3f(this.location("uColor"), m.color[0], m.color[1], m.color[2]);
    gl.uniform1f(this.location("uUseTexture"), textured);
    gl.uniform1f(this.location("uJoints"), view.joints ? 1 : 0);
    gl.uniform1f(this.location("uTurned"), view.turned ? 1 : 0);
    gl.uniform1f(this.location("uGloss"), m.gloss);
    gl.uniform1f(this.location("uMetal"), m.metal);
    gl.uniform1f(this.location("uSparkle"), m.sparkle);
    gl.uniform1f(this.location("uMirror"), m.mirror);
    gl.uniform1f(this.location("uRelief"), m.relief);
    gl.uniform1f(this.location("uSplit"), view.split);
    gl.uniform1f(this.location("uHighlight"), view.highlight);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

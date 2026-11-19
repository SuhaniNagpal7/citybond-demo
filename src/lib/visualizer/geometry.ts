// Wall geometry for the visualizer. Pure functions on typed arrays, so the same code
// runs in the browser worker and in Node tests.
//
// Coordinates: "normalised" image coordinates are centred on the image middle and
// divided by max(W, H). A focal length `f` is in the same unit (f = 0.8 is a phone main camera).

export type Vec3 = [number, number, number];
export type Segment = { x1: number; y1: number; x2: number; y2: number; len: number };
export type Plane = {
  /** Unit wall normal, horizontal wall direction (U) and up direction (V) in camera space. */
  n: Vec3;
  d: Vec3;
  up: Vec3;
  /** Plane offset: n · X = h, in metres after scaling. */
  h: number;
  /** Share of the image this plane covers. */
  area: number;
  /** True when straight lines in the photo confirmed the direction. */
  fromLines: boolean;
};
export type SceneFrame = {
  W: number;
  H: number;
  f: number;
  up: Vec3;
  e1: Vec3;
  e2: Vec3;
  /** Vertical vanishing point (homogeneous, normalised coordinates). */
  v: Vec3;
  /** Wall directions of the scene, as angles around the up axis (radians). The strongest comes first. */
  yaws: number[];
  /** True when straight lines in the photo gave the wall directions. */
  fromLines: boolean;
  /** Inverse depth on a coarse grid, with the unknown shift removed. */
  grid: Float32Array;
  gw: number;
  gh: number;
  step: number;
  /** Metres per depth unit. */
  metres: number;
  scaleFromGround: boolean;
  /** Near-vertical lines: where each line crosses the middle row, its vertical extent and its length. */
  verticals: { s: number; y1: number; y2: number; len: number }[];
  debug: { segments: number; verticalSupport: number; shift: number };
};
export type Faces = {
  /** Per pixel index into `planes`. 255 means the pixel is outside the region. */
  labels: Uint8Array;
  planes: Plane[];
  /** One point on each plane (camera space, metres). Used to turn a plane by hand. */
  anchors: Vec3[];
};

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const median = (values: number[] | Float32Array) => {
  if (!values.length) return 0;
  const sorted = Float32Array.from(values).sort();
  return sorted[sorted.length >> 1];
};

/** Line segment detector: grows regions of pixels that share a gradient direction (the LSD idea). */
export function detectSegments(gray: Float32Array, W: number, H: number): Segment[] {
  const N = W * H;
  const blur = new Float32Array(N);
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      blur[i] =
        (gray[i] * 4 + (gray[i - 1] + gray[i + 1] + gray[i - W] + gray[i + W]) * 2 + gray[i - W - 1] + gray[i - W + 1] + gray[i + W - 1] + gray[i + W + 1]) / 16;
    }
  const mag = new Float32Array(N);
  const ang = new Float32Array(N);
  const threshold = 5.2;
  let maxMag = 0;
  for (let y = 1; y < H - 2; y++)
    for (let x = 1; x < W - 2; x++) {
      const i = y * W + x;
      const gx = (blur[i + 1] + blur[i + W + 1] - blur[i] - blur[i + W]) / 2;
      const gy = (blur[i + W] + blur[i + W + 1] - blur[i] - blur[i + 1]) / 2;
      const m = Math.hypot(gx, gy);
      mag[i] = m;
      ang[i] = Math.atan2(gx, -gy);
      if (m > maxMag) maxMag = m;
    }
  // Strong gradients first.
  const bins = 256;
  const counts = new Int32Array(bins + 1);
  const binOf = (m: number) => Math.min(bins - 1, Math.floor((m / (maxMag || 1)) * bins));
  for (let i = 0; i < N; i++) if (mag[i] > threshold) counts[binOf(mag[i])]++;
  const starts = new Int32Array(bins + 1);
  for (let b = bins - 1; b >= 0; b--) starts[b] = starts[b + 1] + counts[b + 1];
  const order = new Int32Array(starts[0] + counts[0]);
  const cursor = starts.slice();
  for (let i = 0; i < N; i++) if (mag[i] > threshold) order[cursor[binOf(mag[i])]++] = i;

  const used = new Uint8Array(N);
  const stack = new Int32Array(N);
  const region = new Int32Array(N);
  const tolerance = Math.PI / 8;
  const minLength = Math.max(18, 0.022 * Math.max(W, H));
  const out: Segment[] = [];
  const angleDiff = (a: number, b: number) => {
    let d = Math.abs(a - b) % (2 * Math.PI);
    if (d > Math.PI) d = 2 * Math.PI - d;
    return d;
  };
  for (let o = 0; o < order.length; o++) {
    const seed = order[o];
    if (used[seed]) continue;
    let count = 0;
    let top = 0;
    let sumCos = Math.cos(ang[seed]);
    let sumSin = Math.sin(ang[seed]);
    let regionAngle = ang[seed];
    stack[top++] = seed;
    used[seed] = 1;
    while (top) {
      const p = stack[--top];
      region[count++] = p;
      const px = p % W;
      const py = (p - px) / W;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const qx = px + dx;
          const qy = py + dy;
          if (qx < 1 || qy < 1 || qx >= W - 1 || qy >= H - 1) continue;
          const q = qy * W + qx;
          if (used[q] || mag[q] <= threshold || angleDiff(ang[q], regionAngle) > tolerance) continue;
          used[q] = 1;
          stack[top++] = q;
          sumCos += Math.cos(ang[q]);
          sumSin += Math.sin(ang[q]);
          regionAngle = Math.atan2(sumSin, sumCos);
        }
    }
    if (count < minLength) continue;
    let sw = 0;
    let cx = 0;
    let cy = 0;
    for (let k = 0; k < count; k++) {
      const p = region[k];
      const w = mag[p];
      sw += w;
      cx += w * (p % W);
      cy += w * Math.floor(p / W);
    }
    cx /= sw;
    cy /= sw;
    let ixx = 0;
    let iyy = 0;
    let ixy = 0;
    for (let k = 0; k < count; k++) {
      const p = region[k];
      const w = mag[p];
      const dx = (p % W) - cx;
      const dy = Math.floor(p / W) - cy;
      ixx += w * dx * dx;
      iyy += w * dy * dy;
      ixy += w * dx * dy;
    }
    const theta = 0.5 * Math.atan2(2 * ixy, ixx - iyy);
    const ux = Math.cos(theta);
    const uy = Math.sin(theta);
    let lo = 1e9;
    let hi = -1e9;
    let wlo = 1e9;
    let whi = -1e9;
    for (let k = 0; k < count; k++) {
      const p = region[k];
      const dx = (p % W) - cx;
      const dy = Math.floor(p / W) - cy;
      const along = dx * ux + dy * uy;
      const across = -dx * uy + dy * ux;
      if (along < lo) lo = along;
      if (along > hi) hi = along;
      if (across < wlo) wlo = across;
      if (across > whi) whi = across;
    }
    const len = hi - lo;
    const width = whi - wlo + 1;
    if (len < minLength || width > 0.12 * len + 4 || count < 0.5 * len) continue;
    out.push({ x1: cx + lo * ux, y1: cy + lo * uy, x2: cx + hi * ux, y2: cy + hi * uy, len });
  }
  return out;
}

function smallestEigenvector(m: number[]): Vec3 {
  // Jacobi rotations for a symmetric 3x3 matrix [xx, xy, xz, yy, yz, zz].
  const a = [
    [m[0], m[1], m[2]],
    [m[1], m[3], m[4]],
    [m[2], m[4], m[5]],
  ];
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 24; sweep++)
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(a[p][q]) < 1e-14) continue;
      const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p];
        const vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  let best = 0;
  for (let i = 1; i < 3; i++) if (a[i][i] < a[best][best]) best = i;
  return norm([v[0][best], v[1][best], v[2][best]]);
}

type Line = { a: number; b: number; c: number; mx: number; my: number; dx: number; dy: number; len: number };

export function toLines(segments: Segment[], W: number, H: number): Line[] {
  const s = 1 / Math.max(W, H);
  return segments.map((g) => {
    const x1 = (g.x1 - W / 2) * s;
    const y1 = (g.y1 - H / 2) * s;
    const x2 = (g.x2 - W / 2) * s;
    const y2 = (g.y2 - H / 2) * s;
    const l = Math.hypot(x2 - x1, y2 - y1) || 1;
    const dx = (x2 - x1) / l;
    const dy = (y2 - y1) / l;
    const a = -dy;
    const b = dx;
    return { a, b, c: -(a * x1 + b * y1), mx: (x1 + x2) / 2, my: (y1 + y2) / 2, dx, dy, len: l };
  });
}

/** Angle (radians) between a segment and the direction from its middle to a homogeneous point. */
function deviation(line: Line, v: Vec3) {
  const tx = v[0] - line.mx * v[2];
  const ty = v[1] - line.my * v[2];
  const l = Math.hypot(tx, ty);
  if (l < 1e-9) return Math.PI / 2;
  return Math.asin(Math.min(1, Math.abs(tx * line.dy - ty * line.dx) / l));
}

export function verticalVanishingPoint(lines: Line[]): { v: Vec3; support: number } {
  const candidates = lines
    .filter((l) => Math.abs(l.dy) > Math.cos((22 * Math.PI) / 180))
    .sort((p, q) => q.len - p.len)
    .slice(0, 70);
  const level: Vec3 = [0, 1, 0];
  if (candidates.length < 4) return { v: level, support: 0 };
  const tolerance = (2 * Math.PI) / 180;
  const score = (v: Vec3) => candidates.reduce((sum, l) => sum + (deviation(l, v) < tolerance ? l.len : 0), 0);
  let best = level;
  let bestScore = score(level);
  for (let i = 0; i < candidates.length; i++)
    for (let j = i + 1; j < candidates.length; j++) {
      const p = candidates[i];
      const q = candidates[j];
      const v = norm(cross([p.a, p.b, p.c], [q.a, q.b, q.c]));
      // The vertical vanishing point must be far above or below the picture.
      if (Math.abs(v[1]) < 1.2 * Math.abs(v[2]) || Math.abs(v[0]) > Math.abs(v[1])) continue;
      const s = score(v);
      if (s > bestScore) {
        bestScore = s;
        best = v;
      }
    }
  const inliers = candidates.filter((l) => deviation(l, best) < tolerance);
  if (inliers.length >= 3) {
    const m = [0, 0, 0, 0, 0, 0];
    for (const l of inliers) {
      m[0] += l.len * l.a * l.a;
      m[1] += l.len * l.a * l.b;
      m[2] += l.len * l.a * l.c;
      m[3] += l.len * l.b * l.b;
      m[4] += l.len * l.b * l.c;
      m[5] += l.len * l.c * l.c;
    }
    const refined = smallestEigenvector(m);
    if (score(refined) >= bestScore * 0.98) best = refined;
  }
  const total = candidates.reduce((sum, l) => sum + l.len, 0);
  return { v: best, support: bestScore / (total || 1) };
}

function frame(v: Vec3, f: number) {
  let up = norm([v[0], v[1], f * v[2]]);
  if (up[1] > 0) up = [-up[0], -up[1], -up[2]];
  let e1 = norm(cross(up, [0, 0, 1]));
  if (e1[0] < 0) e1 = [-e1[0], -e1[1], -e1[2]];
  const e2 = norm(cross(e1, up));
  return { up, e1, e2 };
}

const BINS = 180;
export function smoothCircular(hist: Float32Array, sigma: number) {
  const out = new Float32Array(BINS);
  const r = Math.ceil(sigma * 3);
  for (let i = 0; i < BINS; i++) {
    let s = 0;
    for (let k = -r; k <= r; k++) s += hist[(i + k + BINS * 4) % BINS] * Math.exp((-k * k) / (2 * sigma * sigma));
    out[i] = s;
  }
  return out;
}
export function peaks(hist: Float32Array, minShare: number, separation: number, limit: number) {
  const max = hist.reduce((m, v) => Math.max(m, v), 0);
  const found: { bin: number; value: number }[] = [];
  for (let i = 0; i < BINS; i++) {
    const v = hist[i];
    if (v < max * minShare || v <= 0) continue;
    if (v >= hist[(i + 1) % BINS] && v >= hist[(i + BINS - 1) % BINS]) found.push({ bin: i, value: v });
  }
  found.sort((p, q) => q.value - p.value);
  const kept: { bin: number; value: number }[] = [];
  for (const p of found) {
    if (kept.length >= limit) break;
    if (kept.every((k) => circularDistance(k.bin, p.bin) >= separation)) kept.push(p);
  }
  return kept;
}
const circularDistance = (a: number, b: number) => {
  const d = Math.abs(a - b) % BINS;
  return Math.min(d, BINS - d);
};

export function yawHistogram(lines: Line[], v: Vec3, f: number) {
  const { up, e1, e2 } = frame(v, f);
  const hist = new Float32Array(BINS);
  const verticalTolerance = (4 * Math.PI) / 180;
  for (const l of lines) {
    if (deviation(l, v) < verticalTolerance) continue;
    const dir = cross([f * l.a, f * l.b, l.c], up);
    const length = Math.hypot(dir[0], dir[1], dir[2]);
    // A line near the horizon gives an unstable direction, so it counts less.
    const stability = Math.min(1, (length / Math.hypot(f * l.a, f * l.b, l.c) - 0.04) / 0.2);
    if (stability <= 0) continue;
    const yaw = (Math.atan2(dot(dir, e2), dot(dir, e1)) + Math.PI * 2) % Math.PI;
    hist[Math.floor((yaw / Math.PI) * BINS) % BINS] += l.len * stability;
  }
  return hist;
}

export type DepthInput = {
  /** Relative inverse depth (larger is nearer), resampled to W x H. */
  disparity: Float32Array;
  /** 0..1 probability that a pixel is a paintable wall. */
  wall: Float32Array;
  /** 1 where the pixel is floor, road or ground. */
  ground: Uint8Array;
  /** 1 where the pixel is sky. */
  sky: Uint8Array;
};

/**
 * Find the camera and the two wall directions of a photo.
 * Straight lines give the precise directions. Depth says which direction a wall area has.
 */
export function analyseScene(gray: Float32Array, W: number, H: number, input: DepthInput): SceneFrame {
  const segments = detectSegments(gray, W, H);
  const lines = toLines(segments, W, H);
  const { v, support } = verticalVanishingPoint(lines);

  // Search the focal length that makes wall directions meet at right angles.
  // The score is the share of line length that has a partner direction 90 degrees away.
  const defaultFocal = 0.78;
  let f = defaultFocal;
  let bestScore = 0.1;
  for (let k = 0; k < 18; k++) {
    const candidate = 0.5 * Math.pow(2.2 / 0.5, k / 17);
    const hist = smoothCircular(yawHistogram(lines, v, candidate), 1.5);
    let pair = 0;
    let self = 0;
    for (let i = 0; i < BINS; i++) {
      pair += hist[i] * hist[(i + 90) % BINS];
      self += hist[i] * hist[i];
    }
    const prior = Math.exp(-Math.pow(Math.log(candidate / defaultFocal), 2) / (2 * 0.45 * 0.45));
    const s = (pair / (self || 1)) * (0.4 + 0.6 * prior);
    if (s > bestScore) {
      bestScore = s;
      f = candidate;
    }
  }
  const { up, e1, e2 } = frame(v, f);
  const hist = smoothCircular(yawHistogram(lines, v, f), 2);
  let total = 0;
  for (let i = 0; i < BINS; i++) total += hist[i];
  let bestBin = 0;
  let bestPair = -1;
  for (let i = 0; i < BINS; i++) {
    const value = Math.max(hist[i], hist[(i + 90) % BINS]) + 0.6 * Math.min(hist[i], hist[(i + 90) % BINS]);
    if (value > bestPair) {
      bestPair = value;
      bestBin = i;
    }
  }
  const fromLines = lines.length >= 12 && bestPair > (total / BINS) * 4;

  // Coarse inverse depth grid.
  const step = 4;
  const gw = Math.floor(W / step);
  const gh = Math.floor(H / step);
  const grid = new Float32Array(gw * gh);
  const gridWall = new Float32Array(gw * gh);
  const gridGround = new Float32Array(gw * gh);
  const gridSky = new Float32Array(gw * gh);
  for (let gy = 0; gy < gh; gy++)
    for (let gx = 0; gx < gw; gx++) {
      let s = 0;
      let w = 0;
      let g = 0;
      let k = 0;
      for (let y = gy * step; y < gy * step + step; y++)
        for (let x = gx * step; x < gx * step + step; x++) {
          const i = y * W + x;
          s += input.disparity[i];
          w += input.wall[i];
          g += input.ground[i];
          k += input.sky[i];
        }
      const n = step * step;
      grid[gy * gw + gx] = s / n;
      gridWall[gy * gw + gx] = w / n;
      gridGround[gy * gw + gx] = g / n;
      gridSky[gy * gw + gx] = k / n;
    }
  const scale = 1 / Math.max(W, H);
  const nx = (gx: number) => ((gx + 0.5) * step - W / 2) * scale;
  const ny = (gy: number) => ((gy + 0.5) * step - H / 2) * scale;

  // The model gives inverse depth up to an unknown shift. Find the shift from the sky or the ground.
  let shift = 0;
  const skyValues: number[] = [];
  const ground = new PlaneFit();
  let groundCells = 0;
  for (let gy = 0; gy < gh; gy += 2)
    for (let gx = 0; gx < gw; gx += 2) {
      const i = gy * gw + gx;
      if (gridSky[i] > 0.9) skyValues.push(grid[i]);
      if (gridGround[i] > 0.9) {
        groundCells++;
        ground.add(nx(gx), ny(gy), grid[i]);
      }
    }
  const cells = (gw * gh) / 4;
  let minDisparity = Infinity;
  for (let i = 0; i < grid.length; i++) if (grid[i] < minDisparity) minDisparity = grid[i];
  const groundFit = groundCells > cells * 0.03 ? ground.solve() : null;
  if (skyValues.length > cells * 0.03) shift = median(skyValues);
  else if (groundFit) {
    // The ground plane must have the up direction as its normal.
    const horizontal = up[0] * up[0] + up[1] * up[1];
    shift = groundFit.c - (f * (groundFit.a * up[0] + groundFit.b * up[1]) * up[2]) / horizontal;
  }
  // Keep every depth positive.
  let maxDisparity = -Infinity;
  for (let i = 0; i < grid.length; i++) if (grid[i] > maxDisparity) maxDisparity = grid[i];
  shift = Math.min(shift, minDisparity - 0.03 * (maxDisparity - minDisparity) - 1e-6);
  if (!Number.isFinite(shift)) shift = 0;
  for (let i = 0; i < grid.length; i++) grid[i] -= shift;

  // Metric scale: the camera is about 1.5 m above the ground.
  let metres = 0;
  let scaleFromGround = false;
  if (groundCells > cells * 0.03) {
    const heights: number[] = [];
    for (let gy = 0; gy < gh; gy += 2)
      for (let gx = 0; gx < gw; gx += 2) {
        const i = gy * gw + gx;
        if (gridGround[i] > 0.9) heights.push(Math.abs(dot(up, [nx(gx) / f, ny(gy) / f, 1])) / grid[i]);
      }
    const height = median(heights);
    if (height > 1e-6 && Number.isFinite(height)) {
      metres = 1.5 / height;
      scaleFromGround = true;
    }
  }
  if (!metres) {
    // No ground in view: assume the walls are about 4 m away.
    const wallDepths: number[] = [];
    for (let i = 0; i < grid.length; i += 3) if (gridWall[i] > 0.9) wallDepths.push(1 / grid[i]);
    metres = wallDepths.length ? 4 / median(wallDepths) : 1;
  }

  // Wall directions. Without lines, use the direction of the largest wall area from depth.
  let yawA = ((bestBin + 0.5) / BINS) * Math.PI;
  if (!fromLines) {
    const all = new PlaneFit();
    for (let gy = 0; gy < gh; gy += 2) for (let gx = 0; gx < gw; gx += 2) if (gridWall[gy * gw + gx] > 0.9) all.add(nx(gx), ny(gy), grid[gy * gw + gx]);
    const fitted = all.solve();
    const yaw = fitted ? yawOfFit(fitted, f, up, e1, e2) : null;
    yawA = yaw ?? 0;
  }
  // The main pair of directions, then other strong line directions (walls that are not at 90 degrees).
  const yaws = [yawA, (yawA + Math.PI / 2) % Math.PI];
  if (fromLines)
    for (const peak of peaks(hist, 0.3, 14, 5)) {
      const yaw = ((peak.bin + 0.5) / BINS) * Math.PI;
      if (yaws.length < 4 && yaws.every((known) => yawDistance(known, yaw) > (14 * Math.PI) / 180)) yaws.push(yaw);
    }
  const verticals = lines
    .map((l, index) => ({ l, g: segments[index] }))
    .filter(({ l }) => deviation(l, v) < (2.5 * Math.PI) / 180)
    .map(({ g }) => ({ s: columnOf(v, W, H, (g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2), y1: Math.min(g.y1, g.y2), y2: Math.max(g.y1, g.y2), len: g.len }));

  return {
    W,
    H,
    f,
    up,
    e1,
    e2,
    v,
    yaws,
    fromLines,
    grid,
    gw,
    gh,
    step,
    metres,
    scaleFromGround,
    verticals,
    debug: { segments: segments.length, verticalSupport: support, shift },
  };
}

/** Least squares fit of d = a x + b y + c. */
class PlaneFit {
  n = 0;
  private sx = 0;
  private sy = 0;
  private sd = 0;
  private sxx = 0;
  private sxy = 0;
  private syy = 0;
  private sxd = 0;
  private syd = 0;
  private sdd = 0;
  /** Sum of squared errors of the best fit. Large when the points do not lie on one plane. */
  error() {
    const n = this.n;
    if (n < 12) return 0;
    const mx = this.sx / n;
    const my = this.sy / n;
    const md = this.sd / n;
    const cxx = this.sxx / n - mx * mx;
    const cxy = this.sxy / n - mx * my;
    const cyy = this.syy / n - my * my;
    const cxd = this.sxd / n - mx * md;
    const cyd = this.syd / n - my * md;
    const det = cxx * cyy - cxy * cxy;
    const variance = this.sdd / n - md * md;
    if (Math.abs(det) < 1e-14) return n * variance;
    const a = (cxd * cyy - cyd * cxy) / det;
    const b = (cyd * cxx - cxd * cxy) / det;
    return Math.max(0, n * (variance - a * cxd - b * cyd));
  }
  add(x: number, y: number, d: number) {
    this.sdd += d * d;
    this.n++;
    this.sx += x;
    this.sy += y;
    this.sd += d;
    this.sxx += x * x;
    this.sxy += x * y;
    this.syy += y * y;
    this.sxd += x * d;
    this.syd += y * d;
  }
  solve() {
    const n = this.n;
    if (n < 12) return null;
    const mx = this.sx / n;
    const my = this.sy / n;
    const md = this.sd / n;
    const cxx = this.sxx / n - mx * mx;
    const cxy = this.sxy / n - mx * my;
    const cyy = this.syy / n - my * my;
    const cxd = this.sxd / n - mx * md;
    const cyd = this.syd / n - my * md;
    const det = cxx * cyy - cxy * cxy;
    if (Math.abs(det) < 1e-14) return null;
    const a = (cxd * cyy - cyd * cxy) / det;
    const b = (cyd * cxx - cxd * cxy) / det;
    return { a, b, c: md - a * mx - b * my };
  }
}

/** Wall direction (angle around the up axis, 0..PI) of a fitted inverse depth plane. Null for floors and ceilings. */
function yawOfFit(p: { a: number; b: number; c: number }, f: number, up: Vec3, e1: Vec3, e2: Vec3) {
  const n = norm([f * p.a, f * p.b, p.c]);
  const tilt = dot(n, up);
  if (Math.abs(tilt) > 0.75) return null;
  const flat = norm([n[0] - tilt * up[0], n[1] - tilt * up[1], n[2] - tilt * up[2]]);
  const dir = cross(up, flat);
  return (Math.atan2(dot(dir, e2), dot(dir, e1)) + Math.PI * 2) % Math.PI;
}

/** Where the vertical line through a pixel crosses the middle row of the picture (pixels). */
function columnOf(v: Vec3, W: number, H: number, x: number, y: number) {
  const size = Math.max(W, H);
  const cy = H / 2;
  if (Math.abs(v[2]) < 1e-6 * Math.abs(v[1])) return x + ((cy - y) * v[0]) / v[1];
  const px = (v[0] / v[2]) * size + W / 2;
  const py = (v[1] / v[2]) * size + H / 2;
  return x + ((cy - y) * (px - x)) / (py - y);
}

const yawDistance = (a: number, b: number) => {
  const d = Math.abs(a - b) % Math.PI;
  return Math.min(d, Math.PI - d);
};

/**
 * Split one selected region into flat wall faces.
 * Walls meet at vertical corners, so the split is a set of vertical boundaries.
 */
export function facesForRegion(scene: SceneFrame, region: Uint8Array): Faces {
  const { W, H, f, up, e1, e2, v, grid, gw, gh, step } = scene;
  const scale = 1 / Math.max(W, H);
  const cellsX: number[] = [];
  const cellsY: number[] = [];
  const cellsS: number[] = [];
  const cellsD: number[] = [];
  let top = H;
  let bottom = 0;
  for (let gy = 0; gy < gh; gy++)
    for (let gx = 0; gx < gw; gx++) {
      const x = gx * step + step / 2;
      const y = gy * step + step / 2;
      if (!region[Math.floor(y) * W + Math.floor(x)]) continue;
      cellsX.push((x - W / 2) * scale);
      cellsY.push((y - H / 2) * scale);
      cellsS.push(columnOf(v, W, H, x, y));
      cellsD.push(grid[gy * gw + gx]);
      if (y < top) top = y;
      if (y > bottom) bottom = y;
    }
  const labels = new Uint8Array(W * H).fill(255);
  if (cellsX.length < 12) return { labels, planes: [], anchors: [] };

  let sMin = Infinity;
  let sMax = -Infinity;
  for (const s of cellsS) {
    if (s < sMin) sMin = s;
    if (s > sMax) sMax = s;
  }
  // Direction along the region, in windows of columns.
  const stride = 8;
  const half = Math.max(36, 0.06 * W);
  const count = Math.max(1, Math.ceil((sMax - sMin) / stride));
  const order = cellsS.map((_, i) => i).sort((p, q) => cellsS[p] - cellsS[q]);
  const side = new Int8Array(count); // 0 = first direction, 1 = second, -1 = unknown
  let lo = 0;
  let hi = 0;
  for (let k = 0; k < count; k++) {
    const centre = sMin + (k + 0.5) * stride;
    while (lo < order.length && cellsS[order[lo]] < centre - half) lo++;
    while (hi < order.length && cellsS[order[hi]] <= centre + half) hi++;
    const fit = new PlaneFit();
    const skip = Math.max(1, Math.floor((hi - lo) / 600));
    for (let j = lo; j < hi; j += skip) fit.add(cellsX[order[j]], cellsY[order[j]], cellsD[order[j]]);
    const solved = fit.n >= 24 ? fit.solve() : null;
    const yaw = solved ? yawOfFit(solved, f, up, e1, e2) : null;
    side[k] = -1;
    if (yaw !== null) {
      let nearest = Infinity;
      scene.yaws.forEach((candidate, index) => {
        // The main pair of directions wins unless another direction is clearly nearer.
        const distance = yawDistance(yaw, candidate) + (index > 1 ? (6 * Math.PI) / 180 : 0);
        if (distance < nearest) {
          nearest = distance;
          side[k] = index;
        }
      });
    }
  }
  // Fill unknown windows from their neighbours, then remove short runs.
  for (let k = 1; k < count; k++) if (side[k] < 0) side[k] = side[k - 1];
  for (let k = count - 2; k >= 0; k--) if (side[k] < 0) side[k] = side[k + 1];
  if (side[0] < 0) side.fill(0);
  const minRun = Math.max(3, Math.round((0.07 * W) / stride));
  for (let pass = 0; pass < 4; pass++) {
    let start = 0;
    for (let k = 1; k <= count; k++) {
      if (k < count && side[k] === side[start]) continue;
      if (k - start < minRun && (start > 0 || k < count)) {
        const replacement = start > 0 ? side[start - 1] : side[k];
        for (let j = start; j < k; j++) side[j] = replacement;
      }
      start = k;
    }
  }
  // Boundaries between runs, moved to the nearest strong vertical line in the photo.
  const boundaries: number[] = [];
  for (let k = 1; k < count; k++) {
    if (side[k] === side[k - 1]) continue;
    // Candidates: the first guess and the vertical lines near it.
    // The best candidate is the one where the depth on each side fits one flat wall best.
    const guess = sMin + k * stride;
    const candidates = [guess];
    for (const line of scene.verticals) {
      if (line.y2 < top || line.y1 > bottom || Math.abs(line.s - guess) > half) continue;
      if (Math.min(line.y2, bottom) - Math.max(line.y1, top) > 0.08 * (bottom - top)) candidates.push(line.s);
    }
    let from = 0;
    while (from < order.length && cellsS[order[from]] < guess - 2 * half) from++;
    let to = from;
    while (to < order.length && cellsS[order[to]] <= guess + 2 * half) to++;
    const skip = Math.max(1, Math.floor((to - from) / 1500));
    let s = guess;
    let lowest = Infinity;
    for (const candidate of candidates) {
      const left = new PlaneFit();
      const right = new PlaneFit();
      for (let j = from; j < to; j += skip) (cellsS[order[j]] < candidate ? left : right).add(cellsX[order[j]], cellsY[order[j]], cellsD[order[j]]);
      const cost = left.error() + right.error();
      if (cost < lowest) {
        lowest = cost;
        s = candidate;
      }
    }
    boundaries.push(s);
  }
  boundaries.sort((p, q) => p - q);
  const faceSide: number[] = [side[0]];
  for (let k = 1; k < count; k++) if (side[k] !== side[k - 1]) faceSide.push(side[k]);

  const faceOf = (s: number) => {
    let i = 0;
    while (i < boundaries.length && s >= boundaries[i]) i++;
    return i;
  };
  const offsets: number[][] = faceSide.map(() => []);
  const centres = faceSide.map(() => [0, 0, 0]);
  const directions = faceSide.map((sideIndex) => {
    const angle = scene.yaws[sideIndex];
    let d: Vec3 = norm([
      Math.cos(angle) * e1[0] + Math.sin(angle) * e2[0],
      Math.cos(angle) * e1[1] + Math.sin(angle) * e2[1],
      Math.cos(angle) * e1[2] + Math.sin(angle) * e2[2],
    ]);
    if (d[0] < 0) d = [-d[0], -d[1], -d[2]];
    let n = norm(cross(d, up));
    if (n[2] > 0) n = [-n[0], -n[1], -n[2]];
    return { d, n };
  });
  const sample = Math.max(1, Math.floor(cellsX.length / 3000));
  let sampled = 0;
  for (let i = 0; i < cellsX.length; i += sample) {
    const face = faceOf(cellsS[i]);
    const ray: Vec3 = [cellsX[i] / f, cellsY[i] / f, 1];
    offsets[face].push(dot(directions[face].n, ray) / cellsD[i]);
    centres[face][0] += ray[0];
    centres[face][1] += ray[1];
    centres[face][2] += 1;
    sampled++;
  }
  const planes: Plane[] = directions.map(({ d, n }, i) => ({
    n,
    d,
    up,
    h: (offsets[i].length ? median(offsets[i]) : -4 / scene.metres) * scene.metres,
    area: offsets[i].length / Math.max(1, sampled),
    fromLines: scene.fromLines,
  }));
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (region[i]) labels[i] = boundaries.length ? faceOf(columnOf(v, W, H, x, y)) : 0;
    }
  const anchors = planes.map((plane, i): Vec3 => {
    const count = centres[i][2] || 1;
    const ray: Vec3 = [centres[i][0] / count, centres[i][1] / count, 1];
    const t = plane.h / dot(plane.n, ray);
    return [ray[0] * t, ray[1] * t, t];
  });
  return { labels, planes, anchors };
}

/** Turn a plane around the up axis, keeping the point `anchor` (camera space, metres) on the plane. */
export function rotatePlane(plane: Plane, anchor: Vec3, radians: number): Plane {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  const k = plane.up;
  const kxd = cross(k, plane.d);
  const kd = dot(k, plane.d) * (1 - c);
  let d: Vec3 = norm([plane.d[0] * c + kxd[0] * s + k[0] * kd, plane.d[1] * c + kxd[1] * s + k[1] * kd, plane.d[2] * c + kxd[2] * s + k[2] * kd]);
  if (d[0] < 0) d = [-d[0], -d[1], -d[2]];
  let n = norm(cross(d, plane.up));
  if (n[2] > 0) n = [-n[0], -n[1], -n[2]];
  return { ...plane, d, n, h: dot(n, anchor) };
}

/** Texture coordinates in metres for a pixel on a plane. Mirrors the shader. */
export function planeUV(plane: Plane, f: number, x: number, y: number, W: number, H: number): [number, number] {
  const s = 1 / Math.max(W, H);
  const ray: Vec3 = [((x - W / 2) * s) / f, ((y - H / 2) * s) / f, 1];
  const t = plane.h / dot(plane.n, ray);
  return [t * dot(ray, plane.d), -t * dot(ray, plane.up)];
}

/**
 * Shading layer: how light or dark each wall pixel is compared with the wall's own base tone.
 * An edge-keeping blur removes the old wall texture but keeps shadows.
 */
export function shadingMap(gray: Float32Array, wall: Float32Array, W: number, H: number): { map: Float32Array; w: number; h: number } {
  const step = 2;
  const w = Math.floor(W / step);
  const h = Math.floor(H / step);
  let lum = new Float32Array(w * h);
  const mask = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      let m = 0;
      for (let dy = 0; dy < step; dy++)
        for (let dx = 0; dx < step; dx++) {
          const i = (y * step + dy) * W + x * step + dx;
          s += gray[i];
          m += wall[i];
        }
      lum[y * w + x] = s / (step * step) / 255;
      mask[y * w + x] = m / (step * step);
    }
  const radius = 5;
  const sigmaRange = 0.09;
  for (let pass = 0; pass < 2; pass++) {
    const next = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const centre = lum[i];
        let s = 0;
        let ws = 0;
        for (let dy = -radius; dy <= radius; dy += 1) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -radius; dx <= radius; dx += 1) {
            const xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            const j = yy * w + xx;
            const diff = lum[j] - centre;
            const weight = (0.05 + mask[j]) * Math.exp(-(diff * diff) / (2 * sigmaRange * sigmaRange));
            s += weight * lum[j];
            ws += weight;
          }
        }
        next[i] = s / ws;
      }
    lum = next;
  }
  return { map: lum, w, h };
}

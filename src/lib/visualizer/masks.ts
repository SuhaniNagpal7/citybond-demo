// Mask helpers for the wall visualizer. Pure functions on typed arrays: they run in the worker and in Node tests.

export function resample(source: Float32Array, sw: number, sh: number, W: number, H: number) {
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const fy = Math.min(sh - 1, Math.max(0, ((y + 0.5) * sh) / H - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ay = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(sw - 1, Math.max(0, ((x + 0.5) * sw) / W - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const ax = fx - x0;
      out[y * W + x] = (source[y0 * sw + x0] * (1 - ax) + source[y0 * sw + x1] * ax) * (1 - ay) + (source[y1 * sw + x0] * (1 - ax) + source[y1 * sw + x1] * ax) * ay;
    }
  }
  return out;
}

/** Connected area of `allowed` pixels around a start pixel. */
export function flood(allowed: (i: number) => boolean, W: number, H: number, start: number) {
  const mask = new Uint8Array(W * H);
  if (start < 0 || !allowed(start)) return mask;
  const queue = new Int32Array(W * H);
  let head = 0;
  let tail = 0;
  queue[tail++] = start;
  mask[start] = 1;
  while (head < tail) {
    const p = queue[head++];
    const x = p % W;
    if (x > 0 && !mask[p - 1] && allowed(p - 1)) {
      mask[p - 1] = 1;
      queue[tail++] = p - 1;
    }
    if (x < W - 1 && !mask[p + 1] && allowed(p + 1)) {
      mask[p + 1] = 1;
      queue[tail++] = p + 1;
    }
    if (p >= W && !mask[p - W] && allowed(p - W)) {
      mask[p - W] = 1;
      queue[tail++] = p - W;
    }
    if (p < W * (H - 1) && !mask[p + W] && allowed(p + W)) {
      mask[p + W] = 1;
      queue[tail++] = p + W;
    }
  }
  return mask;
}

/** Erode (every pixel in the window is set) or dilate (any pixel in the window is set) a mask. */
export function windowTest(mask: Uint8Array, W: number, H: number, radius: number, every: boolean) {
  const integral = new Int32Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) {
      row += mask[y * W + x];
      integral[(y + 1) * (W + 1) + x + 1] = integral[y * (W + 1) + x + 1] + row;
    }
  }
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(H, y + radius + 1);
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(W, x + radius + 1);
      const sum = integral[y1 * (W + 1) + x1] - integral[y0 * (W + 1) + x1] - integral[y1 * (W + 1) + x0] + integral[y0 * (W + 1) + x0];
      out[y * W + x] = (every ? sum === (y1 - y0) * (x1 - x0) : sum > 0) ? 1 : 0;
    }
  }
  return out;
}

/** Remove ragged edges and pin holes from a mask. */
export function tidy(mask: Uint8Array, W: number, H: number, radius: number) {
  const integral = new Int32Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) {
      row += mask[y * W + x];
      integral[(y + 1) * (W + 1) + x + 1] = integral[y * (W + 1) + x + 1] + row;
    }
  }
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(H, y + radius + 1);
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(W, x + radius + 1);
      const sum = integral[y1 * (W + 1) + x1] - integral[y0 * (W + 1) + x1] - integral[y1 * (W + 1) + x0] + integral[y0 * (W + 1) + x0];
      out[y * W + x] = sum * 2 > (y1 - y0) * (x1 - x0) ? 1 : 0;
    }
  }
  return out;
}

/**
 * Take windows and other openings out of a building wall.
 * An opening is a small dark area that the wall surrounds. A shadow is large or touches the wall edge, so it stays.
 */
export function removeOpenings(region: Uint8Array, gray: Float32Array, W: number, H: number) {
  const sumGray = new Float64Array((W + 1) * (H + 1));
  const sumMask = new Int32Array((W + 1) * (H + 1));
  let area = 0;
  for (let y = 0; y < H; y++) {
    let rowGray = 0;
    let rowMask = 0;
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (region[i]) {
        rowGray += gray[i];
        rowMask++;
        area++;
      }
      sumGray[(y + 1) * (W + 1) + x + 1] = sumGray[y * (W + 1) + x + 1] + rowGray;
      sumMask[(y + 1) * (W + 1) + x + 1] = sumMask[y * (W + 1) + x + 1] + rowMask;
    }
  }
  const radius = Math.round(0.06 * Math.max(W, H));
  const box = (table: Float64Array | Int32Array, x: number, y: number, r: number) => {
    const x0 = Math.max(0, x - r);
    const x1 = Math.min(W, x + r + 1);
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(H, y + r + 1);
    return table[y1 * (W + 1) + x1] - table[y0 * (W + 1) + x1] - table[y1 * (W + 1) + x0] + table[y0 * (W + 1) + x0];
  };
  const dark = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!region[i]) continue;
      const local = box(sumGray, x, y, radius) / Math.max(1, box(sumMask, x, y, radius));
      if (gray[i] < 0.66 * local - 4) dark[i] = 1;
    }
  const seen = new Uint8Array(W * H);
  const queue = new Int32Array(W * H);
  for (let start = 0; start < W * H; start++) {
    if (!dark[start] || seen[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    seen[start] = 1;
    let touchesEdge = 0;
    let border = 0;
    while (head < tail) {
      const p = queue[head++];
      const x = p % W;
      const neighbours = [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p >= W ? p - W : -1, p < W * (H - 1) ? p + W : -1];
      for (const q of neighbours) {
        if (q < 0) {
          touchesEdge++;
          border++;
        } else if (dark[q]) {
          if (!seen[q]) {
            seen[q] = 1;
            queue[tail++] = q;
          }
        } else {
          border++;
          if (!region[q]) touchesEdge++;
        }
      }
    }
    // Small, and the wall surrounds most of it.
    if (tail < 30 || tail > 0.06 * area || touchesEdge > 0.3 * border) continue;
    for (let k = 0; k < tail; k++) {
      const p = queue[k];
      const px = p % W;
      const py = (p - px) / W;
      // Remove the area and a thin ring around it (the frame).
      for (let dy = -3; dy <= 3; dy++)
        for (let dx = -3; dx <= 3; dx++) {
          const qx = px + dx;
          const qy = py + dy;
          if (qx >= 0 && qy >= 0 && qx < W && qy < H) region[qy * W + qx] = 0;
        }
    }
  }
}

/** Mean of each pixel's square window. Windows are cut at the picture border. */
function boxMean(source: Float32Array, W: number, H: number, radius: number) {
  const temp = new Float32Array(W * H);
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    let sum = 0;
    const row = y * W;
    for (let x = 0; x < Math.min(W, radius); x++) sum += source[row + x];
    for (let x = 0; x < W; x++) {
      if (x + radius < W) sum += source[row + x + radius];
      if (x - radius - 1 >= 0) sum -= source[row + x - radius - 1];
      temp[row + x] = sum / (Math.min(W - 1, x + radius) - Math.max(0, x - radius) + 1);
    }
  }
  for (let x = 0; x < W; x++) {
    let sum = 0;
    for (let y = 0; y < Math.min(H, radius); y++) sum += temp[y * W + x];
    for (let y = 0; y < H; y++) {
      if (y + radius < H) sum += temp[(y + radius) * W + x];
      if (y - radius - 1 >= 0) sum -= temp[(y - radius - 1) * W + x];
      out[y * W + x] = sum / (Math.min(H - 1, y + radius) - Math.max(0, y - radius) + 1);
    }
  }
  return out;
}

/**
 * Guided filter (He, Sun, Tang). It moves the soft edge of `mask` onto the real edges of the photo.
 * `guide` is the photo brightness, 0..1. A larger `radius` lets an edge move further.
 */
export function guidedFilter(mask: Float32Array, guide: Float32Array, W: number, H: number, radius: number, epsilon: number) {
  const N = W * H;
  const product = new Float32Array(N);
  const square = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    product[i] = guide[i] * mask[i];
    square[i] = guide[i] * guide[i];
  }
  const meanGuide = boxMean(guide, W, H, radius);
  const meanMask = boxMean(mask, W, H, radius);
  const meanProduct = boxMean(product, W, H, radius);
  const meanSquare = boxMean(square, W, H, radius);
  const a = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const variance = meanSquare[i] - meanGuide[i] * meanGuide[i];
    const covariance = meanProduct[i] - meanGuide[i] * meanMask[i];
    a[i] = covariance / (variance + epsilon);
    b[i] = meanMask[i] - a[i] * meanGuide[i];
  }
  const meanA = boxMean(a, W, H, radius);
  const meanB = boxMean(b, W, H, radius);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = Math.min(1, Math.max(0, meanA[i] * guide[i] + meanB[i]));
  return out;
}

/**
 * Pick up to `count` points that are far from each other, from the pixels where `allowed` is true.
 * `first` is always the first point, when it is given.
 */
export function spreadPoints(allowed: (i: number) => boolean, W: number, H: number, count: number, first?: number) {
  const candidates: number[] = [];
  for (let y = 8; y < H; y += 16) for (let x = 8; x < W; x += 16) if (allowed(y * W + x)) candidates.push(y * W + x);
  const picked: number[] = first !== undefined ? [first] : candidates.length ? [candidates[candidates.length >> 1]] : [];
  const minimum = Math.pow(0.055 * Math.max(W, H), 2);
  while (picked.length < count && candidates.length) {
    let best = -1;
    let bestDistance = -1;
    for (const c of candidates) {
      let nearest = Infinity;
      for (const q of picked) {
        const dx = (c % W) - (q % W);
        const dy = Math.floor(c / W) - Math.floor(q / W);
        nearest = Math.min(nearest, dx * dx + dy * dy);
      }
      if (nearest > bestDistance) {
        bestDistance = nearest;
        best = c;
      }
    }
    if (bestDistance < minimum) break;
    picked.push(best);
  }
  return picked;
}

/** The largest connected areas of a mask, each as its own mask. Areas smaller than `minArea` pixels are skipped. */
export function largestAreas(mask: Uint8Array, W: number, H: number, minArea: number, limit: number) {
  const seen = new Uint8Array(W * H);
  const found: { area: number; mask: Uint8Array }[] = [];
  for (let y = 4; y < H; y += 8)
    for (let x = 4; x < W; x += 8) {
      const start = y * W + x;
      if (!mask[start] || seen[start]) continue;
      const area = flood((i) => mask[i] === 1, W, H, start);
      let size = 0;
      for (let i = 0; i < area.length; i++)
        if (area[i]) {
          seen[i] = 1;
          size++;
        }
      if (size >= minArea) found.push({ area: size, mask: area });
    }
  return found.sort((p, q) => q.area - p.area).slice(0, limit);
}

/**
 * Move the edge of a mask onto the real edges of the photo: a wide step finds the edge, a narrow step makes it crisp.
 * Use it where the wall has a clear contrast with its surroundings, for example a building against the sky.
 */
export function snapToEdges(mask: Uint8Array, guide: Float32Array, W: number, H: number) {
  const wide = guidedFilter(Float32Array.from(mask), guide, W, H, Math.round(0.012 * Math.max(W, H)), 4e-4);
  for (let i = 0; i < wide.length; i++) {
    const t = Math.min(1, Math.max(0, (wide[i] - 0.3) / 0.4));
    wide[i] = t * t * (3 - 2 * t);
  }
  const narrow = guidedFilter(wide, guide, W, H, 3, 1e-4);
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = narrow[i] > 0.5 ? 1 : 0;
  return out;
}

/**
 * Fill small holes that the selected wall surrounds, for example the shadow of a leaf on the wall.
 * `wall` is the wall probability. A hole that borders other things (a picture in its frame) stays open.
 */
export function fillEnclosedHoles(mask: Uint8Array, wall: Float32Array, W: number, H: number, maxArea: number) {
  const seen = new Uint8Array(W * H);
  const queue = new Int32Array(W * H);
  for (let start = 0; start < W * H; start++) {
    if (mask[start] || seen[start] || wall[start] <= 0.5) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    seen[start] = 1;
    let border = 0;
    let onWall = 0;
    while (head < tail) {
      const p = queue[head++];
      const x = p % W;
      const neighbours = [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p >= W ? p - W : -1, p < W * (H - 1) ? p + W : -1];
      for (const q of neighbours) {
        if (q < 0) border++;
        else if (mask[q]) {
          border++;
          onWall++;
        } else if (wall[q] <= 0.5) border++;
        else if (!seen[q]) {
          seen[q] = 1;
          queue[tail++] = q;
        }
      }
    }
    if (tail <= maxArea && onWall >= 0.8 * border) for (let k = 0; k < tail; k++) mask[queue[k]] = 1;
  }
}

/** Opacity (1..255) for the pixels of a region, with a soft, anti-aliased edge taken from the refined mask. */
export function regionAlpha(region: Uint8Array, refined: Float32Array, W: number, H: number) {
  const near = windowTest(region, W, H, 2, false);
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) {
    if (!near[i]) continue;
    const t = Math.min(1, Math.max(0, (refined[i] - 0.3) / 0.4));
    const value = Math.round(255 * t * t * (3 - 2 * t));
    out[i] = region[i] ? Math.max(1, value) : value;
  }
  return out;
}

import assert from "node:assert/strict";
import test from "node:test";
import { fillEnclosedHoles, guidedFilter, largestAreas, removeOpenings, snapToEdges, spreadPoints } from "../src/lib/visualizer/masks";

const W = 120;
const H = 80;

test("snapToEdges moves a rough mask edge onto the edge in the photo", () => {
  // The search distance grows with the picture size, so this test uses a wide picture.
  const width = 600;
  const height = 60;
  // Photo: bright wall on the left of column 300, dark object on the right.
  const guide = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) guide[y * width + x] = x < 300 ? 0.8 : 0.2;
  // Rough mask: the edge is 4 pixels too far left.
  const rough = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < 296; x++) rough[y * width + x] = 1;
  const snapped = snapToEdges(rough, guide, width, height);
  const row = 30 * width;
  let edge = 0;
  while (edge < width && snapped[row + edge]) edge++;
  assert.ok(Math.abs(edge - 300) <= 1, `edge at ${edge}, expected 300`);
});

test("guidedFilter keeps a mask that already follows the photo", () => {
  const guide = new Float32Array(W * H);
  const mask = new Float32Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      guide[y * W + x] = x < 60 ? 0.9 : 0.1;
      mask[y * W + x] = x < 60 ? 1 : 0;
    }
  const out = guidedFilter(mask, guide, W, H, 4, 1e-4);
  assert.ok(out[40 * W + 58] > 0.95 && out[40 * W + 61] < 0.05);
});

test("fillEnclosedHoles fills a hole inside the wall but not a picture in a frame", () => {
  const wall = new Float32Array(W * H).fill(1);
  const mask = new Uint8Array(W * H).fill(1);
  // Hole 1: a leaf shadow. The selected wall surrounds it.
  for (let y = 20; y < 26; y++) for (let x = 20; x < 28; x++) mask[y * W + x] = 0;
  // Hole 2: a picture mat. A frame (not wall) surrounds it.
  for (let y = 40; y < 60; y++)
    for (let x = 60; x < 90; x++) {
      mask[y * W + x] = 0;
      const frame = y < 43 || y >= 57 || x < 63 || x >= 87;
      if (frame) wall[y * W + x] = 0;
    }
  fillEnclosedHoles(mask, wall, W, H, 400);
  assert.equal(mask[22 * W + 24], 1, "the shadow hole is filled");
  assert.equal(mask[50 * W + 75], 0, "the picture stays open");
});

test("removeOpenings takes a dark window out of a building wall and keeps a large shadow", () => {
  const gray = new Float32Array(W * H).fill(220);
  const region = new Uint8Array(W * H).fill(1);
  for (let y = 30; y < 42; y++) for (let x = 50; x < 62; x++) gray[y * W + x] = 40; // window
  for (let y = 0; y < H; y++) for (let x = 0; x < 30; x++) gray[y * W + x] = 110; // shadow that reaches the wall edge
  removeOpenings(region, gray, W, H);
  assert.equal(region[36 * W + 56], 0, "the window is out");
  assert.equal(region[40 * W + 10], 1, "the shadow stays covered");
  assert.equal(region[10 * W + 90], 1);
});

test("spreadPoints and largestAreas", () => {
  const mask = new Uint8Array(W * H);
  for (let y = 4; y < 40; y++) for (let x = 4; x < 60; x++) mask[y * W + x] = 1;
  for (let y = 56; y < 76; y++) for (let x = 84; x < 116; x++) mask[y * W + x] = 1;
  const areas = largestAreas(mask, W, H, 100, 5);
  assert.equal(areas.length, 2);
  assert.ok(areas[0].area > areas[1].area);
  const first = 20 * W + 20;
  const points = spreadPoints((i) => mask[i] === 1, W, H, 6, first);
  assert.equal(points[0], first);
  assert.ok(points.length >= 2 && points.every((i) => mask[i] === 1));
});

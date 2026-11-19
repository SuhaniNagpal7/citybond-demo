import assert from "node:assert/strict";
import test from "node:test";
import { analyseScene, facesForRegion, planeUV, rotatePlane, type Plane, type Vec3 } from "../src/lib/visualizer/geometry";

const frontal: Plane = { n: [0, 0, -1], d: [1, 0, 0], up: [0, -1, 0], h: -4, area: 1, fromLines: true };

test("planeUV gives metres on a wall that faces the camera", () => {
  const W = 1000;
  const H = 500;
  const f = 1;
  // The wall is 4 m away. With f = 1, the picture is 4 m wide at that distance.
  const [u0, v0] = planeUV(frontal, f, 500, 250, W, H);
  const [u1, v1] = planeUV(frontal, f, 750, 250, W, H);
  const [, v2] = planeUV(frontal, f, 500, 0, W, H);
  assert.ok(Math.abs(u0) < 1e-9 && Math.abs(v0) < 1e-9);
  assert.ok(Math.abs(u1 - 1) < 1e-9 && Math.abs(v1) < 1e-9);
  assert.ok(Math.abs(v2 + 1) < 1e-9, "V grows downwards, like the rows of the sheet image");
});

test("rotatePlane keeps the anchor point on the plane", () => {
  const anchor: Vec3 = [0.5, -0.2, 4];
  const turned = rotatePlane(frontal, anchor, (30 * Math.PI) / 180);
  const onPlane = turned.n[0] * anchor[0] + turned.n[1] * anchor[1] + turned.n[2] * anchor[2];
  assert.ok(Math.abs(onPlane - turned.h) < 1e-9);
  assert.ok(Math.abs(Math.hypot(...turned.n) - 1) < 1e-9);
  assert.ok(Math.abs(turned.n[0] * turned.d[0] + turned.n[1] * turned.d[1] + turned.n[2] * turned.d[2]) < 1e-9);
  assert.ok(turned.n[2] < 0, "the wall still faces the camera");
});

test("a room corner is split into two faces at the corner", () => {
  // Synthetic room: a back wall 4 m away on the left, a side wall that comes towards the camera on the right.
  const W = 400;
  const H = 300;
  const f = 0.8;
  const size = Math.max(W, H);
  const corner = 260;
  const gray = new Float32Array(W * H);
  const disparity = new Float32Array(W * H);
  const wall = new Float32Array(W * H).fill(1);
  const cornerX = (corner - W / 2) / size / f;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const rx = (x - W / 2) / size / f;
      const ry = (y - H / 2) / size / f;
      let depth = 4;
      let u: number;
      if (x >= corner) {
        // Side wall: the plane X = cornerX * 4, so depth = cornerX * 4 / rx.
        depth = (cornerX * 4) / rx;
        u = 4 - depth;
      } else u = rx * 4;
      const v = ry * depth;
      // Tile lines every 0.5 m give the detector straight lines on both walls.
      const line = Math.abs(u / 0.5 - Math.round(u / 0.5)) < 0.04 || Math.abs(v / 0.5 - Math.round(v / 0.5)) < 0.04;
      gray[y * W + x] = line ? 40 : x >= corner ? 150 : 200;
      disparity[y * W + x] = 1 / depth;
    }
  const scene = analyseScene(gray, W, H, { disparity, wall, ground: new Uint8Array(W * H), sky: new Uint8Array(W * H) });
  const region = new Uint8Array(W * H).fill(1);
  const faces = facesForRegion(scene, region);
  assert.equal(faces.planes.length, 2);
  const left = faces.labels[150 * W + 60];
  const right = faces.labels[150 * W + 380];
  assert.notEqual(left, right);
  // The boundary is near the true corner.
  let boundary = 0;
  for (let x = 1; x < W; x++) if (faces.labels[150 * W + x] !== faces.labels[150 * W + x - 1]) boundary = x;
  assert.ok(Math.abs(boundary - corner) <= 12, `corner found at ${boundary}, expected ${corner}`);
  // The back wall faces the camera. The side wall is at a right angle to the back wall.
  assert.ok(Math.abs(faces.planes[left].n[2]) > 0.97);
  assert.ok(Math.abs(faces.planes[right].n[0]) > 0.97);
});

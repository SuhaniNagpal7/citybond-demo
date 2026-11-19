import type { Product } from "@/lib/cms/schema";
import type { Material } from "./renderer";

type Look = Partial<Pick<Material, "texture" | "gloss" | "metal" | "sparkle" | "mirror" | "relief">> & { tile?: [number, number]; finish?: string; color?: string };

// A standard CityBond sheet is 1220 mm x 2440 mm. The sheet images in /public/citybond are full sheets.
const SHEET: [number, number] = [1.22, 2.44];

/** How each product is drawn. Products that are not listed here have no texture yet and are not offered. */
const LOOKS: Record<string, Look> = {
  "lamio-series": { texture: "/citybond/lamio-swatch.webp", gloss: 0.15 },
  "exclusive-series": { texture: "/citybond/exclusive-swatch.webp", gloss: 0.1 },
  "wooden-series": { texture: "/citybond/wooden-swatch.webp", gloss: 0.12 },
  "marble-series": { texture: "/citybond/marble-swatch.webp", gloss: 0.4 },
  "metallic-series": { metal: 0.9, sparkle: 0.35, gloss: 0.3 },
  "satin-series": { metal: 0.6, gloss: 0.15 },
  "solid-series": { gloss: 0.2 },
  "gloss-series": { gloss: 0.9 },
  "sparkle-series": { texture: "/citybond/sparkle-swatch.webp", sparkle: 1, gloss: 0.5 },
  "mirror-series": { mirror: 0.7, gloss: 0.8 },
  // Fluted 3D panel with a stone print. The ribs are drawn by the shader.
  "3d-panels-series": { texture: "/citybond/marble-swatch.webp", relief: 1, gloss: 0.1, finish: "Fluted stone", color: "#d1c9ae" },
};

const rgb = (hex: string): [number, number, number] => {
  const value = /^#?([0-9a-f]{6})$/i.exec(hex)?.[1] || "b8b8b8";
  // sRGB values are used as they are: the photo is not linearised either.
  return [parseInt(value.slice(0, 2), 16) / 255, parseInt(value.slice(2, 4), 16) / 255, parseInt(value.slice(4, 6), 16) / 255];
};

export function materialsFrom(products: Product[]): Material[] {
  return products.flatMap((product) => {
    const look = LOOKS[product.slug];
    if (!look) return [];
    const finish = product.finishes[0];
    const hex = finish?.hex || look.color || "#b8b8b8";
    return [
      {
        slug: product.slug,
        name: product.name,
        finish: finish?.name || look.finish || product.collection,
        texture: look.texture || "",
        color: rgb(hex),
        tile: look.tile || SHEET,
        gloss: look.gloss || 0,
        metal: look.metal || 0,
        sparkle: look.sparkle || 0,
        mirror: look.mirror || 0,
        relief: look.relief || 0,
      },
    ];
  });
}

export const swatchStyle = (material: Material) =>
  material.texture
    ? { backgroundImage: `url(${material.texture})`, backgroundSize: "cover" }
    : { background: `linear-gradient(135deg, rgb(${material.color.map((v) => Math.round(Math.min(1, v * 1.25) * 255)).join(",")}), rgb(${material.color.map((v) => Math.round(v * 255)).join(",")}) 55%, rgb(${material.color.map((v) => Math.round(v * 0.72 * 255)).join(",")}))` };

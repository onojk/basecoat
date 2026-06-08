//! basecoat GUI — thin viewer/driver over the headless engine.
//! egui/eframe + wgpu backend.

use basecoat::adjust::invert as adjust_invert;
use basecoat::bands::{generate_bands, growth_to_thickness};
use basecoat::edge::{aggr_to_n, edge};
use basecoat::kaleido::kaleido;
use basecoat::layers::*;
use basecoat::plasma::{apply_plasma_with_progress, H as IMG_H, W as IMG_W};
use basecoat::qbist::{create_info, optimize, render_with_progress as qbist_render_threaded};
use basecoat::spiral::{
    render_with_progress as spiral_render_threaded, SpiralKind, OVERSAMPLING as SPIRAL_OS,
};
use basecoat::quad::quad as quad_fn;
use basecoat::selection::{mask_coverage_pct, select_by_color, union_masks};
use basecoat::punch::punch;
use eframe::egui;
use png::{BitDepth, ColorType, Encoder, Unit};
use std::collections::HashSet;
use std::io::BufWriter;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    mpsc, Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};

const W: u32 = IMG_W as u32;
const H: u32 = IMG_H as u32;
const PPM: u32 = 23622;

const THUMB_SIZE: usize = 64;  // texture resolution
const THUMB_PX:   f32   = 40.0; // display size in the layer row

// Marching-ants tuning — change these to adjust outline appearance.
const ANT_THICKNESS:  f32 = 2.5;  // minimum mark size in screen pixels (tune: 1.0 = hairline)
const DASH_LEN:       i32 = 14;   // white-dash length in screen pixels
const DASH_GAP:       i32 = 14;   // black-gap length in screen pixels
// Feather-band guide lines (inner 0.9 / outer 0.1 contours drawn faint + static).
const GUIDE_THICKNESS: f32 = 1.5;  // screen px for guide-line squares
const GUIDE_GREY:      u8  = 160;  // guide-line luminance (0=black, 255=white)
const GUIDE_ALPHA:     u8  = 140;  // guide-line opacity  (255=opaque)
// Coverage feedback
const HIGH_COVERAGE_WARN_PCT: f64 = 85.0;  // % above which a pick is rejected
const DEFAULT_AGGR:           f32 = 25.0;  // default Aggressiveness slider value
const DEFAULT_FEATHER:        f32 = 15.0;  // default Feather slider value
// Power-curve exponents that back-load the Aggr/Feather sliders.
// pct_effective = (pct/100)^GAMMA * 100, so low/mid values stay precise
// and the broad range is pushed toward the top of the slider travel.
// Both default to 3.0; use separate consts for independent tuning later.
const AGGR_GAMMA:    f32 = 3.0;
const FEATHER_GAMMA: f32 = 3.0;

// Payload type for the background render channel.
// Plasma sends linear-light f32 directly; qbist sends sRGB u8 for conversion.
enum RenderBuf {
    /// Already linear-light — copy straight into layer.rgba.
    Linear(Vec<f32>),
    /// sRGB u8 — apply srgb_to_linear before writing into layer.rgba.
    SrgbU8(Vec<u8>),
}

// ---------------------------------------------------------------------------
// Export helper (same encode path as headless binaries)
// ---------------------------------------------------------------------------

fn export_png(layer: &Layer, path: &std::path::Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let bw = BufWriter::new(file);
    let mut enc = Encoder::new(bw, W, H);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions {
        xppu: PPM,
        yppu: PPM,
        unit: Unit::Meter,
    }));
    let mut writer = enc.write_header().map_err(|e| std::io::Error::other(e.to_string()))?;
    let mut buf = vec![0u8; (W * H * 4) as usize];
    for i in (0..layer.rgba.len()).step_by(4) {
        buf[i    ] = srgb_enc(layer.rgba[i    ]);
        buf[i + 1] = srgb_enc(layer.rgba[i + 1]);
        buf[i + 2] = srgb_enc(layer.rgba[i + 2]);
        buf[i + 3] = (layer.rgba[i + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    writer.write_image_data(&buf).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}

#[inline]
fn srgb_enc(v: f32) -> u8 {
    (linear_to_srgb_f64(v as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Canvas composite → egui ColorImage
// ---------------------------------------------------------------------------

fn layer_to_color_image(layer: &Layer) -> egui::ColorImage {
    let w = layer.width as usize;
    let h = layer.height as usize;
    let mut pixels = Vec::with_capacity(w * h);
    for i in (0..layer.rgba.len()).step_by(4) {
        pixels.push(egui::Color32::from_rgba_unmultiplied(
            srgb_enc(layer.rgba[i    ]),
            srgb_enc(layer.rgba[i + 1]),
            srgb_enc(layer.rgba[i + 2]),
            (layer.rgba[i + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ));
    }
    egui::ColorImage { size: [w, h], pixels }
}

/// Extract boundary pixels (0.5 contour) from a selection mask.
/// A pixel is on the boundary if it differs (>= 0.5 vs < 0.5) from any 4-neighbor.
/// Stored in image coords as (col, row) for fast projection to screen space.
/// Only recomputed when the mask changes — cached between frames.
/// Pixels that sit on the threshold crossing at `thresh` (4-neighbour test).
/// Returns image-space (col, row) pairs for all boundary pixels.
fn compute_selection_boundary(mask: &[f32], w: usize, h: usize, thresh: f32) -> Vec<(u16, u16)> {
    let mut out = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let i   = row * w + col;
            let sel = mask[i] >= thresh;
            let on_edge =
                (row > 0     && (mask[(row - 1) * w + col    ] >= thresh) != sel) ||
                (row + 1 < h && (mask[(row + 1) * w + col    ] >= thresh) != sel) ||
                (col > 0     && (mask[ row      * w + col - 1] >= thresh) != sel) ||
                (col + 1 < w && (mask[ row      * w + col + 1] >= thresh) != sel);
            if on_edge {
                out.push((col as u16, row as u16));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Thumbnail: box-average downsample + checkerboard composite
// ---------------------------------------------------------------------------

/// Generate the thumbnail pixel data without touching any egui texture state.
/// The caller is responsible for uploading via ctx.load_texture or tex.set().
fn make_thumb_image(layer: &Layer) -> egui::ColorImage {
    let lw = layer.width  as usize;
    let lh = layer.height as usize;
    let mut pixels = Vec::with_capacity(THUMB_SIZE * THUMB_SIZE);

    for ty in 0..THUMB_SIZE {
        let y_lo = ty       * lh / THUMB_SIZE;
        let y_hi = (ty + 1) * lh / THUMB_SIZE;
        let y_hi = y_hi.max(y_lo + 1).min(lh); // at least 1 row

        for tx in 0..THUMB_SIZE {
            let x_lo = tx       * lw / THUMB_SIZE;
            let x_hi = (tx + 1) * lw / THUMB_SIZE;
            let x_hi = x_hi.max(x_lo + 1).min(lw);

            // Box-average in linear space
            let (mut sr, mut sg, mut sb, mut sa) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let count = ((y_hi - y_lo) * (x_hi - x_lo)) as f64;
            for y in y_lo..y_hi {
                for x in x_lo..x_hi {
                    let i = (y * lw + x) * 4;
                    sr += layer.rgba[i    ] as f64;
                    sg += layer.rgba[i + 1] as f64;
                    sb += layer.rgba[i + 2] as f64;
                    sa += layer.rgba[i + 3] as f64;
                }
            }
            let (r, g, b, a) = (
                (sr / count) as f32,
                (sg / count) as f32,
                (sb / count) as f32,
                (sa / count) as f32,
            );

            // Convert linear → sRGB u8
            let rs = srgb_enc(r);
            let gs = srgb_enc(g);
            let bs = srgb_enc(b);

            // Checkerboard: 8-px squares, #c0c0c0 (192) / #808080 (128)
            let ck = if ((tx / 8) + (ty / 8)) % 2 == 0 { 192u8 } else { 128u8 };

            // Straight-alpha composite over checkerboard (in sRGB u8)
            let blend = |ch: u8| -> u8 {
                (a * ch as f32 + (1.0 - a) * ck as f32 + 0.5) as u8
            };
            pixels.push(egui::Color32::from_rgb(blend(rs), blend(gs), blend(bs)));
        }
    }

    egui::ColorImage { size: [THUMB_SIZE, THUMB_SIZE], pixels }
}

// ---------------------------------------------------------------------------
// Time-based seed for "New Seed"
// ---------------------------------------------------------------------------

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345)
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct BasecoatApp {
    stack:  Stack,
    active: usize,

    // Canvas composite cache
    dirty:   bool,
    texture: Option<egui::TextureHandle>,

    // Per-layer thumbnail cache, parallel to stack.layers.
    // thumb_dirty[i] = true means the texture needs regeneration.
    // Regenerated lazily, one per frame, to avoid per-frame stutter.
    // Invariant maintained by sync_thumb_cache(): len == stack.layers.len().
    thumb_textures: Vec<Option<egui::TextureHandle>>,
    thumb_dirty:    Vec<bool>,

    // Multi-select mark set (indices into stack.layers).
    // Policy: cleared on all structural operations (add/delete/reorder/
    // flatten/undo/band_generate) to prevent stale index references.
    marked: HashSet<usize>,

    // Technique controls
    plasma_seed:        u64,
    plasma_seed_str:    String,
    plasma_turbulence:  f32,

    qbist_seed:         u64,
    qbist_seed_str:     String,
    qbist_oversampling: u8,

    spiral_kind:    SpiralKind,
    spiral_turns:   u32,
    spiral_arms:    u32,
    spiral_color_a: [u8; 3],
    spiral_color_b: [u8; 3],

    // Background render state — None when idle.
    render_rx:       Option<mpsc::Receiver<RenderBuf>>,
    render_progress: Arc<AtomicU32>,
    render_total_rows: usize,
    render_target_idx: usize,
    render_finish:   String,
    render_label:    String,

    punch_contrast:   f32,
    punch_saturation: f32,
    punch_passes:     u32,

    band_growth: f32,
    band_fill:   f32,

    edge_aggr:  f32,
    edge_width: usize,

    quad_tiles: u32,

    kaleido_segments: i32,
    kaleido_rotation: f32,
    kaleido_zoom:     f32,

    // Select by Color  (selection_mask + has_selection live in self.stack)
    selection_aggr:         f32,
    selection_feather:      f32,
    selection_pick_mode:    bool,
    selection_coverage_pct: f64,
    // Three cached contours (image coords) for the feather-band visualisation.
    // All recomputed together only when the mask changes; never touched during draw.
    selection_boundary:       Vec<(u16, u16)>,  // main  — mask >= 0.5 (marching ants)
    selection_boundary_inner: Vec<(u16, u16)>,  // inner — mask >= 0.9 (faint guide)
    selection_boundary_outer: Vec<(u16, u16)>,  // outer — mask >= 0.1 (faint guide)
    selection_boundary_dirty: bool,

    // Zoom / pan
    zoom:     f32,
    fit_zoom: f32,
    pan:      egui::Vec2,

    status: String,
}

impl BasecoatApp {
    fn new() -> Self {
        let mut stack = Stack::new();
        // Initialise selection state BEFORE the first stack.add() so the very
        // first structural snap already carries a valid (empty) selection.
        stack.selection_mask = vec![0.0f32; IMG_W * IMG_H];
        stack.has_selection  = false;
        let base = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
        stack.add(base).unwrap();

        Self {
            stack,
            active:  0,
            dirty:   true,
            texture: None,

            thumb_textures: vec![None],
            thumb_dirty:    vec![true],
            marked:         HashSet::new(),

            plasma_seed:       0,
            plasma_seed_str:   "0".into(),
            plasma_turbulence: 1.0,

            qbist_seed:         0,
            qbist_seed_str:     "0".into(),
            qbist_oversampling: 4,

            spiral_kind:    SpiralKind::Logarithmic,
            spiral_turns:   8,
            spiral_arms:    1,
            spiral_color_a: [0,   0,   0  ],   // black (sRGB)
            spiral_color_b: [255, 255, 255],   // white (sRGB)

            render_rx:         None,
            render_progress:   Arc::new(AtomicU32::new(0)),
            render_total_rows: 0,
            render_target_idx: 0,
            render_finish:     String::new(),
            render_label:      String::new(),

            punch_contrast:   9.0,
            punch_saturation: 4.0,
            punch_passes:     6,

            band_growth: 30.0,
            band_fill:   90.0,

            edge_aggr:  50.0,
            edge_width: 3,

            quad_tiles: 2,

            kaleido_segments: 6,
            kaleido_rotation: 0.0,
            kaleido_zoom:     1.0,

            selection_aggr:         DEFAULT_AGGR,
            selection_feather:      DEFAULT_FEATHER,
            selection_pick_mode:    false,
            selection_coverage_pct: 0.0,
            selection_boundary:       Vec::new(),
            selection_boundary_inner: Vec::new(),
            selection_boundary_outer: Vec::new(),
            selection_boundary_dirty: false,

            zoom:     0.0,
            fit_zoom: 1.0,
            pan:      egui::Vec2::ZERO,

            status: "Ready".into(),
        }
    }

    // ---- Thumbnail helpers ------------------------------------------------

    /// Ensure parallel thumb vecs have the same length as the layer stack.
    fn sync_thumb_cache(&mut self) {
        let n = self.stack.layers.len();
        while self.thumb_textures.len() < n {
            self.thumb_textures.push(None);
            self.thumb_dirty.push(true);
        }
        self.thumb_textures.truncate(n);
        self.thumb_dirty.truncate(n);
    }

    /// Mark one slot dirty (pixel content changed, but layer still exists).
    ///
    /// The existing TextureHandle is kept alive intentionally: dropping it
    /// mid-frame while a shape still references its TextureId causes a wgpu
    /// "texture destroyed" validation panic. The handle's content will be
    /// updated in-place via set() at the start of the next frame.
    fn thumb_invalidate(&mut self, i: usize) {
        if i < self.thumb_dirty.len() {
            self.thumb_dirty[i] = true;
            // Do NOT touch thumb_textures[i] — keep the handle alive.
        }
    }

    /// Mark every slot dirty (undo / flatten / new canvas).
    ///
    /// Handles are preserved for the same reason as thumb_invalidate().
    fn thumb_invalidate_all(&mut self) {
        self.sync_thumb_cache();
        for d in &mut self.thumb_dirty { *d = true; }
        // Handles are not dropped here; set() will update them next frame.
    }

    /// Insert a new dirty slot at position `i`.
    fn thumb_insert(&mut self, i: usize) {
        self.thumb_textures.insert(i, None);
        self.thumb_dirty.insert(i, true);
    }

    /// Remove the slot at `i` (layer was deleted).
    fn thumb_remove(&mut self, i: usize) {
        if i < self.thumb_textures.len() {
            self.thumb_textures.remove(i);
            self.thumb_dirty.remove(i);
        }
    }

    /// Regenerate the first dirty thumbnail (at most one per frame).
    ///
    /// If a handle already exists for the slot, its contents are updated
    /// in-place via `TextureHandle::set()` so the TextureId never changes.
    /// This avoids the wgpu "texture destroyed" panic that occurs when a
    /// handle is dropped and recreated within the same frame: egui shapes
    /// capturing the old TextureId would reference a freed GPU texture.
    ///
    /// Calls ctx.request_repaint() when more dirty slots remain.
    fn ensure_thumbnails(&mut self, ctx: &egui::Context) {
        self.sync_thumb_cache();
        for i in 0..self.stack.layers.len() {
            if self.thumb_dirty[i] {
                let img = make_thumb_image(&self.stack.layers[i]);
                match &mut self.thumb_textures[i] {
                    Some(tex) => {
                        // Update existing texture in-place — TextureId stays stable.
                        tex.set(img, egui::TextureOptions::LINEAR);
                    }
                    slot @ None => {
                        // First creation: allocate a new handle.
                        *slot = Some(ctx.load_texture(
                            format!("layer_thumb_{i}"),
                            img,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
                self.thumb_dirty[i] = false;
                if self.thumb_dirty.iter().any(|&d| d) {
                    ctx.request_repaint();
                }
                return;
            }
        }
    }

    // ---- Misc helpers -----------------------------------------------------

    fn ensure_composite(&mut self, ctx: &egui::Context) {
        if !self.dirty || self.stack.layers.is_empty() { return; }
        let comp = self.stack.composite();
        let img  = layer_to_color_image(&comp);
        self.texture = Some(ctx.load_texture("canvas", img, egui::TextureOptions::LINEAR));
        self.dirty   = false;
    }

    fn active_name(&self) -> String {
        self.stack.layers.get(self.active)
            .map(|l| if l.name.is_empty() { format!("Layer {}", self.active) } else { l.name.clone() })
            .unwrap_or_default()
    }

    fn clamp_active(&mut self) {
        let n = self.stack.layers.len();
        if n == 0       { self.active = 0; }
        else if self.active >= n { self.active = n - 1; }
    }

    fn clear_selection(&mut self) {
        self.stack.selection_mask.fill(0.0);
        self.stack.has_selection          = false;
        self.selection_coverage_pct       = 0.0;
        self.selection_boundary.clear();
        self.selection_boundary_inner.clear();
        self.selection_boundary_outer.clear();
        self.selection_boundary_dirty     = false;
        self.dirty                        = true;
    }
}

// ---------------------------------------------------------------------------
// Blend mode helpers
// ---------------------------------------------------------------------------

const MODES: &[BlendMode] = &[
    BlendMode::Normal,
    BlendMode::Multiply,
    BlendMode::Screen,
    BlendMode::Overlay,
    BlendMode::Difference,
];

fn mode_label(m: BlendMode) -> &'static str {
    match m {
        BlendMode::Normal     => "Normal",
        BlendMode::Multiply   => "Multiply",
        BlendMode::Screen     => "Screen",
        BlendMode::Overlay    => "Overlay",
        BlendMode::Difference => "Difference",
    }
}

// ---------------------------------------------------------------------------
// eframe App
// ---------------------------------------------------------------------------

impl eframe::App for BasecoatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ---- Poll background render (plasma / qbist) ------------------------
        // try_recv() borrow ends when `poll` is assigned (owned Result), so self is free after.
        let poll = self.render_rx.as_ref().map(|rx| rx.try_recv());
        match poll {
            Some(Ok(buf)) => {
                self.render_rx = None;
                let idx    = self.render_target_idx;
                let status = std::mem::take(&mut self.render_finish);
                if idx < self.stack.layers.len() {
                    let lw = self.stack.layers[idx].width  as usize;
                    let lh = self.stack.layers[idx].height as usize;
                    let n_px = lw * lh;
                    // Mask blend only when selection active and dims match.
                    let use_mask = self.stack.has_selection && self.stack.selection_mask.len() == n_px;
                    if !use_mask && self.stack.has_selection {
                        self.status = format!(
                            "{}: selection size mismatch — applied to whole layer",
                            self.render_label
                        );
                    }
                    match buf {
                        RenderBuf::SrgbU8(pixels) => {
                            if use_mask {
                                // Convert to linear, then lerp through mask into the existing layer.
                                for i in 0..n_px {
                                    let m = self.stack.selection_mask[i];
                                    let s = i * 4;
                                    let nr = srgb_to_linear(pixels[s    ] as f32 / 255.0);
                                    let ng = srgb_to_linear(pixels[s + 1] as f32 / 255.0);
                                    let nb = srgb_to_linear(pixels[s + 2] as f32 / 255.0);
                                    let layer = &mut self.stack.layers[idx];
                                    layer.rgba[s    ] = layer.rgba[s    ] * (1.0 - m) + nr * m;
                                    layer.rgba[s + 1] = layer.rgba[s + 1] * (1.0 - m) + ng * m;
                                    layer.rgba[s + 2] = layer.rgba[s + 2] * (1.0 - m) + nb * m;
                                    layer.rgba[s + 3] = layer.rgba[s + 3] * (1.0 - m) + m;
                                }
                            } else {
                                let layer = &mut self.stack.layers[idx];
                                for i in 0..n_px {
                                    let s = i * 4;
                                    layer.rgba[s    ] = srgb_to_linear(pixels[s    ] as f32 / 255.0);
                                    layer.rgba[s + 1] = srgb_to_linear(pixels[s + 1] as f32 / 255.0);
                                    layer.rgba[s + 2] = srgb_to_linear(pixels[s + 2] as f32 / 255.0);
                                    layer.rgba[s + 3] = 1.0;
                                }
                            }
                        }
                        RenderBuf::Linear(pixels) => {
                            if use_mask {
                                // Save old pixels, then lerp new → old through mask.
                                let old = self.stack.layers[idx].rgba.clone();
                                for i in 0..n_px {
                                    let m = self.stack.selection_mask[i];
                                    let s = i * 4;
                                    for c in 0..4usize {
                                        self.stack.layers[idx].rgba[s + c] =
                                            old[s + c] * (1.0 - m) + pixels[s + c] * m;
                                    }
                                }
                            } else {
                                self.stack.layers[idx].rgba = pixels;
                            }
                        }
                    }
                    self.dirty = true;
                    self.thumb_invalidate(idx);
                    self.status = status;
                } else {
                    self.status = format!(
                        "{} render finished but target layer was deleted — result discarded",
                        self.render_label
                    );
                }
            }
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                self.render_rx = None;
                self.status = format!("{} render thread disconnected unexpectedly", self.render_label);
            }
            Some(Err(mpsc::TryRecvError::Empty)) | None => {}
        }

        self.ensure_composite(ctx);
        self.ensure_thumbnails(ctx);

        // ---- Keyboard zoom ------------------------------------------------
        const ZOOM_STEP: f32 = 1.25;
        let (zoom_in, zoom_out, zoom_fit) = ctx.input(|i| {
            let ctrl = i.modifiers.ctrl;
            (
                ctrl && (i.key_pressed(egui::Key::Equals) || i.key_pressed(egui::Key::Plus)),
                ctrl && i.key_pressed(egui::Key::Minus),
                ctrl && i.key_pressed(egui::Key::Num0),
            )
        });
        if zoom_in  { let c = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom }; self.zoom = (c * ZOOM_STEP).clamp(0.05, 10.0); }
        if zoom_out { let c = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom }; self.zoom = (c / ZOOM_STEP).clamp(0.05, 10.0); }
        if zoom_fit { self.zoom = 0.0; self.pan = egui::Vec2::ZERO; }

        // ---- Menu bar -----------------------------------------------------
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.stack = Stack::new();
                        self.stack.selection_mask = vec![0.0f32; IMG_W * IMG_H];
                        self.stack.has_selection  = false;
                        let base = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
                        self.stack.add(base).unwrap();
                        self.active               = 0;
                        self.dirty                = true;
                        self.thumb_textures       = vec![None];
                        self.thumb_dirty          = vec![true];
                        self.marked.clear();
                        self.selection_pick_mode      = false;
                        self.selection_coverage_pct   = 0.0;
                        self.selection_boundary.clear();
                        self.selection_boundary_inner.clear();
                        self.selection_boundary_outer.clear();
                        self.selection_boundary_dirty = false;
                        self.status = "New canvas".into();
                        ui.close_menu();
                    }
                    if ui.button("Export PNG…").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new()
                            .set_file_name("export.png")
                            .add_filter("PNG", &["png"])
                            .save_file()
                        {
                            let comp = self.stack.composite();
                            match export_png(&comp, &path) {
                                Ok(())  => self.status = format!("Exported {}", path.display()),
                                Err(e)  => self.status = format!("Export error: {e}"),
                            }
                        }
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo").clicked() {
                        if self.stack.undo() {
                            // Undo can restore the pre-first-layer empty state.
                            // Re-add a blank canvas so the stack is never empty.
                            if self.stack.layers.is_empty() {
                                self.stack.layers.push(Layer::new(W, H, [0.0; 4]));
                                self.thumb_textures = vec![None];
                                self.thumb_dirty    = vec![true];
                                self.active         = 0;
                            }
                            self.clamp_active();
                            self.dirty = true;
                            self.thumb_invalidate_all();
                            self.marked.clear();
                            // Recompute derived selection fields from the restored mask.
                            // stack.selection_mask + stack.has_selection were rolled back
                            // atomically with the pixels inside stack.undo().
                            self.selection_coverage_pct = if self.stack.has_selection {
                                mask_coverage_pct(&self.stack.selection_mask)
                            } else {
                                0.0
                            };
                            self.selection_boundary_dirty = self.stack.has_selection;
                            self.status = "Undo".into();
                        } else {
                            self.status = "Nothing to undo".into();
                        }
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    if ui.button("Zoom In   Ctrl+=").clicked() {
                        let c = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom };
                        self.zoom = (c * ZOOM_STEP).clamp(0.05, 10.0);
                        ui.close_menu();
                    }
                    if ui.button("Zoom Out  Ctrl+\u{2212}").clicked() {
                        let c = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom };
                        self.zoom = (c / ZOOM_STEP).clamp(0.05, 10.0);
                        ui.close_menu();
                    }
                    if ui.button("Fit       Ctrl+0").clicked() {
                        self.zoom = 0.0;
                        self.pan  = egui::Vec2::ZERO;
                        ui.close_menu();
                    }
                });
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            let n_marked = self.marked.len();
            let status = if n_marked > 0 {
                format!("{}  ·  {} marked", self.status, n_marked)
            } else {
                self.status.clone()
            };
            ui.label(status);
        });

        egui::SidePanel::right("layers_panel")
            .min_width(280.0)
            .max_width(340.0)
            .show(ctx, |ui| {
                self.show_layers_panel(ui);
            });

        let ant_time = ctx.input(|i| i.time);
        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_canvas(ui, ant_time);
        });

        // Keep painting while a render is in flight (progress bar) or a selection is
        // active (ants animation). No repaint when truly idle.
        if self.render_rx.is_some() || self.stack.has_selection {
            ctx.request_repaint();
        }
    }
}

impl BasecoatApp {
    fn show_canvas(&mut self, ui: &mut egui::Ui, ant_time: f64) {
        let Some(tex) = &self.texture else { return; };
        let avail    = ui.available_size();
        let img_size = egui::vec2(W as f32, H as f32);
        let fit_zoom = (avail.x / img_size.x).min(avail.y / img_size.y);
        self.fit_zoom = fit_zoom;
        let effective_zoom = if self.zoom == 0.0 { fit_zoom } else { self.zoom };
        let display_size   = img_size * effective_zoom;

        let drag = ui.input(|i| {
            if i.pointer.middle_down() || i.pointer.secondary_down() { i.pointer.delta() }
            else { egui::Vec2::ZERO }
        });
        self.pan += drag;

        // Recompute the boundary pixel cache when the mask has changed.
        // This is a one-shot scan (not per-frame) — only triggered by Pick or Clear.
        if self.selection_boundary_dirty && self.stack.has_selection {
            let mask = &self.stack.selection_mask;
            self.selection_boundary       = compute_selection_boundary(mask, IMG_W, IMG_H, 0.5);
            self.selection_boundary_inner = compute_selection_boundary(mask, IMG_W, IMG_H, 0.9);
            self.selection_boundary_outer = compute_selection_boundary(mask, IMG_W, IMG_H, 0.1);
            self.selection_boundary_dirty = false;
        }

        let pick_mode = self.selection_pick_mode;
        let has_sel   = self.stack.has_selection;
        // Dash phase: advances at 6 image-pixels/second so the ants crawl noticeably.
        let ant_phase = (ant_time * 6.0) as i32;

        // Collect pick result here to avoid borrow conflict when mutating self later.
        let mut pick_result: Option<(Vec<f32>, usize, usize)> = None;

        egui::ScrollArea::both().id_salt("canvas_scroll").show(ui, |ui| {
            let offset_rect = egui::Rect::from_min_size(ui.cursor().min + self.pan, display_size);
            let sense = if pick_mode {
                egui::Sense::click_and_drag()
            } else {
                egui::Sense::drag()
            };
            let resp = ui.allocate_rect(offset_rect, sense);
            // Pan on primary drag only when not in pick mode
            if !pick_mode && resp.dragged_by(egui::PointerButton::Primary) {
                self.pan += resp.drag_delta();
            }
            ui.painter().image(
                tex.id(),
                offset_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Three-contour feather-band display — drawn over the canvas, display-only.
            // Projects image-space pixels to screen, deduplicates per contour, then draws:
            //   outer (mask>=0.1): faint static grey — leading edge of feather zone
            //   main  (mask>=0.5): crisp animated black/white ants — reference boundary
            //   inner (mask>=0.9): faint static grey — fully-selected edge
            // When Feather=0 all three coincide; as Feather grows the band widens.
            if has_sel {
                let painter = ui.painter();

                // Closure: project a boundary list to deduplicated screen points.
                // Captures offset_rect + effective_zoom by copy (both Copy types).
                let project = |boundary: &[(u16, u16)]| -> Vec<egui::Pos2> {
                    let mut seen: HashSet<(i32, i32)> = HashSet::new();
                    let mut pts  = Vec::new();
                    for &(bx, by) in boundary {
                        let sx  = offset_rect.min.x + bx as f32 * effective_zoom;
                        let sy  = offset_rect.min.y + by as f32 * effective_zoom;
                        let key = (sx as i32, sy as i32);
                        if seen.insert(key) {
                            pts.push(egui::pos2(sx, sy));
                        }
                    }
                    pts
                };

                let guide_color = egui::Color32::from_rgba_unmultiplied(
                    GUIDE_GREY, GUIDE_GREY, GUIDE_GREY, GUIDE_ALPHA,
                );
                let guide_sz = effective_zoom.max(GUIDE_THICKNESS);

                // Outer guide (mask >= 0.1): static faint grey.
                if !self.selection_boundary_outer.is_empty() {
                    for pt in project(&self.selection_boundary_outer) {
                        painter.rect_filled(
                            egui::Rect::from_min_size(pt, egui::vec2(guide_sz, guide_sz)),
                            0.0,
                            guide_color,
                        );
                    }
                }

                // Inner guide (mask >= 0.9): static faint grey.
                if !self.selection_boundary_inner.is_empty() {
                    for pt in project(&self.selection_boundary_inner) {
                        painter.rect_filled(
                            egui::Rect::from_min_size(pt, egui::vec2(guide_sz, guide_sz)),
                            0.0,
                            guide_color,
                        );
                    }
                }

                // Main ants (mask >= 0.5): animated black/white dashes (unchanged).
                if !self.selection_boundary.is_empty() {
                    let screen_pts = project(&self.selection_boundary);
                    let px_sz  = effective_zoom.max(ANT_THICKNESS);
                    let period = DASH_LEN + DASH_GAP;
                    for (i, pt) in screen_pts.iter().enumerate() {
                        let slot  = (i as i32 + ant_phase).rem_euclid(period);
                        let color = if slot < DASH_LEN { egui::Color32::WHITE } else { egui::Color32::BLACK };
                        painter.rect_filled(
                            egui::Rect::from_min_size(*pt, egui::vec2(px_sz, px_sz)),
                            0.0,
                            color,
                        );
                    }
                }
            }

            // Pick-mode click: sample color from active layer
            if pick_mode && resp.clicked() {
                if let Some(click_pos) = resp.interact_pointer_pos() {
                    let rel = click_pos - offset_rect.min;
                    let px = (rel.x / effective_zoom).clamp(0.0, W as f32 - 1.0) as usize;
                    let py = (rel.y / effective_zoom).clamp(0.0, H as f32 - 1.0) as usize;
                    // Apply back-loaded power curve so low slider values stay precise.
                    // select_by_color does (pct/100)*MAX_DIST internally; passing the
                    // curved pct gives tol = (raw/100)^GAMMA * MAX_DIST with no change
                    // to selection.rs.
                    let aggr    = (self.selection_aggr    / 100.0).powf(AGGR_GAMMA)    * 100.0;
                    let feather = (self.selection_feather / 100.0).powf(FEATHER_GAMMA) * 100.0;
                    let active  = self.active;
                    if let Some(layer) = self.stack.layers.get(active) {
                        let src = (py * W as usize + px) * 4;
                        if src + 2 < layer.rgba.len() {
                            let pr = layer.rgba[src    ];
                            let pg = layer.rgba[src + 1];
                            let pb = layer.rgba[src + 2];
                            let pick = select_by_color(
                                &layer.rgba, W as usize, H as usize,
                                pr, pg, pb, aggr, feather,
                            );
                            pick_result = Some((pick, px, py));
                        }
                    }
                }
            }
        });

        // Apply the pick (outside ScrollArea / immutable borrow scope).
        // Rejection gate: evaluate the INDIVIDUAL pick's coverage before touching
        // any state. This keeps additive large-but-intentional selections working —
        // only a single over-broad pick is blocked, never the accumulated total.
        if let Some((pick, _px, _py)) = pick_result {
            let pick_cov = mask_coverage_pct(&pick);
            if pick_cov >= HIGH_COVERAGE_WARN_PCT {
                // Reject: wipe any existing selection and reset sliders — full fresh start.
                self.clear_selection();
                self.selection_aggr    = DEFAULT_AGGR;
                self.selection_feather = DEFAULT_FEATHER;
                self.status = format!(
                    "Pick rejected: would select {:.1}% — selection and sliders reset, pick again.",
                    pick_cov,
                );
            } else {
                if self.stack.has_selection {
                    // Additive: grow the existing selection.
                    union_masks(&mut self.stack.selection_mask, &pick);
                } else {
                    // No active selection — start fresh so stale mask bytes can't accumulate.
                    self.stack.selection_mask = pick;
                }
                self.stack.has_selection      = true;
                self.dirty                    = true;
                self.selection_boundary_dirty = true;
                self.selection_coverage_pct   = mask_coverage_pct(&self.stack.selection_mask);
                self.status = format!("Selection: {:.1}%", self.selection_coverage_pct);
            }
        }
    }

    // ---- Band generator ---------------------------------------------------

    fn run_band_generate(&mut self) {
        if self.stack.layers.is_empty() { return; }

        let seed_idx = self.active;
        let growth   = self.band_growth;
        let fill     = self.band_fill;

        let seed_mask: Vec<bool> = {
            let l = &self.stack.layers[seed_idx];
            l.rgba.chunks_exact(4).map(|c| c[3] > 0.5).collect()
        };
        let pw = self.stack.layers[seed_idx].width  as usize;
        let ph = self.stack.layers[seed_idx].height as usize;

        let thickness = growth_to_thickness(growth);
        let bands     = generate_bands(&seed_mask, pw, ph, thickness, fill);
        if bands.is_empty() {
            self.status = "Band Generate: seed has no opaque pixels".into();
            return;
        }

        self.stack.checkpoint();
        self.marked.clear(); // structural change — clear all marks

        let mut gen_count: usize = 0;

        for band in &bands {
            // 12-cap: merge outermost 6 generator layers before adding a 13th
            if gen_count == 12 {
                let mb = seed_idx;
                let total_px = pw * ph;
                let mut rgba = vec![0.0f32; total_px * 4];
                for li in mb..mb + 6 {
                    let l = &self.stack.layers[li];
                    for px in 0..total_px {
                        let b = px * 4;
                        if l.rgba[b + 3] > 0.5 {
                            rgba[b    ] = l.rgba[b    ];
                            rgba[b + 1] = l.rgba[b + 1];
                            rgba[b + 2] = l.rgba[b + 2];
                            rgba[b + 3] = l.rgba[b + 3];
                        }
                    }
                }
                let mut merged  = Layer::new(pw as u32, ph as u32, [0.0; 4]);
                merged.rgba     = rgba;
                merged.name     = "[band]merged".into();
                self.stack.layers[mb] = merged;
                // Drop the 5 now-redundant entries
                for _ in 0..5 {
                    self.stack.layers.remove(mb + 1);
                    self.thumb_textures.remove(mb + 1);
                    self.thumb_dirty.remove(mb + 1);
                }
                self.thumb_textures[mb] = None;
                self.thumb_dirty[mb]    = true;
                self.active    -= 5;
                gen_count       = 7;
            }

            let mut layer = Layer::new(pw as u32, ph as u32, [0.0; 4]);
            let (r, g, b) = if band.white { (1.0f32, 1.0, 1.0) } else { (0.0, 0.0, 0.0) };
            for (px, &in_band) in band.mask.iter().enumerate() {
                if in_band {
                    let base = px * 4;
                    layer.rgba[base    ] = r;
                    layer.rgba[base + 1] = g;
                    layer.rgba[base + 2] = b;
                    layer.rgba[base + 3] = 1.0;
                }
            }
            layer.name = format!("[band]{}", gen_count + 1);

            self.stack.layers.insert(seed_idx, layer);
            self.thumb_insert(seed_idx);
            self.active    += 1;
            gen_count      += 1;
        }

        self.dirty  = true;
        self.status = format!("Generated {gen_count} band layers (growth={growth:.0}% fill={fill:.0}%)");
    }

    // ---- Kaleidoscope apply -----------------------------------------------

    fn run_kaleido(&mut self) {
        if self.stack.layers.is_empty() { return; }

        let segs = self.kaleido_segments as u32;
        let rot  = self.kaleido_rotation as f64;
        let zoom = self.kaleido_zoom     as f64;

        // Collect marked indices (ascending = bottom-first in stack order).
        // self.marked stores STACK INDICES (same coordinate system as self.stack.layers[i]).
        let mut sorted: Vec<usize> = self.marked.iter().copied().collect();
        sorted.sort_unstable();

        eprintln!("[kaleido] marks={:?} active={} stack_len={}",
                  sorted, self.active, self.stack.layers.len());

        // When no layers are marked, kaleido does nothing and shows a hint.
        // (Previously fell back to active layer — removed to prevent silent mis-targeting.)
        if sorted.is_empty() {
            self.status = "Kaleidoscope: no layers marked — check a layer checkbox first".into();
            return;
        }

        // Build (stack_idx, display_name, source_layer) triples — one per marked layer.
        // Force visible=true so a marked-but-hidden layer still gets kaleidoscoped.
        let sources: Vec<(usize, String, Layer)> = sorted.iter().map(|&i| {
            let l = &self.stack.layers[i];
            let name = if l.name.is_empty() { format!("Layer {i}") } else { l.name.clone() };
            let mut src = l.clone();
            src.visible = true;
            eprintln!("[kaleido]   source stack[{i}] = \"{name}\"");
            (i, name, src)
        }).collect();

        let n = sources.len();
        if self.stack.layers.len() + n > MAX_LAYERS {
            self.status = format!(
                "Layer cap ({MAX_LAYERS}) — need {n} slot{} for kaleido",
                if n == 1 { "" } else { "s" }
            );
            return;
        }

        // One undo snapshot for all N outputs; undo removes them all at once.
        self.stack.checkpoint();

        // Append each kaleido output above the current stack top, in source
        // order (bottom source → lowest new output, top source → topmost).
        for (src_idx, src_name, src_layer) in &sources {
            let mut out = kaleido(src_layer, segs, rot, zoom);
            // Embed source stack index in the name so it's unambiguous in the panel.
            out.name = format!("kaleido[{src_idx}]: {src_name}");
            self.stack.layers.push(out);
            self.thumb_textures.push(None);
            self.thumb_dirty.push(true);
        }

        self.active = self.stack.layers.len() - 1;
        self.dirty  = true;
        self.marked.clear();

        // Status: list source names so user can see which layers were processed.
        let src_list: Vec<String> = sources.iter()
            .map(|(i, name, _)| format!("[{i}]{name}"))
            .collect();
        let plural = if n == 1 { "layer" } else { "layers" };
        self.status = format!("Kaleidoscope: {n} {plural} (seg={segs}) — sources: {}",
                              src_list.join(", "));
    }

    // ---- Layers panel -----------------------------------------------------

    fn show_layers_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Layers");
        ui.separator();

        // ---- Layer ops row ----
        ui.horizontal(|ui| {
            let at_cap = self.stack.layers.len() >= MAX_LAYERS;
            if ui.add_enabled(!at_cap, egui::Button::new("+ New"))
                .on_hover_text("New transparent layer (0,0,0,0) inserted above active")
                .clicked()
            {
                if at_cap {
                    self.status = format!("Layer cap ({MAX_LAYERS}) reached — cannot add");
                } else {
                    let insert_at  = self.active + 1;
                    let layer_name = format!("Layer {}", self.stack.layers.len());
                    let mut layer  = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
                    layer.name     = layer_name.clone();
                    // Structural snapshot so the whole insert is one Undo step.
                    self.stack.checkpoint();
                    self.stack.layers.insert(insert_at, layer);
                    self.thumb_insert(insert_at);
                    self.active = insert_at;
                    self.dirty  = true;
                    self.marked.clear();
                    self.status = format!("Added transparent layer \"{layer_name}\"");
                }
            }
            if ui.add_enabled(!at_cap, egui::Button::new("Dup"))
                .on_hover_text("Deep-copy active layer; insert above it, make it active")
                .clicked()
            {
                if at_cap {
                    self.status = format!("Layer cap ({MAX_LAYERS}) reached — cannot duplicate");
                } else {
                    let (dup_layer, copy_name) = {
                        let src  = &self.stack.layers[self.active];
                        let name = if src.name.is_empty() {
                            format!("Layer {} copy", self.active)
                        } else {
                            format!("{} copy", src.name)
                        };
                        let mut d  = Layer::new(src.width, src.height, [0.0; 4]);
                        d.rgba     = src.rgba.clone();
                        d.mode     = src.mode;
                        d.opacity  = src.opacity;
                        d.visible  = src.visible;
                        d.name     = name.clone();
                        (d, name)
                    };
                    let insert_at = self.active + 1;
                    self.stack.checkpoint();
                    self.stack.layers.insert(insert_at, dup_layer);
                    self.thumb_insert(insert_at);
                    self.active = insert_at;
                    self.dirty  = true;
                    self.marked.clear();
                    self.status = format!("Duplicated \"{copy_name}\"");
                }
            }
            if ui.button("Del").on_hover_text("Delete active layer").clicked()
                && !self.stack.layers.is_empty()
            {
                self.stack.remove(self.active);
                self.thumb_remove(self.active);
                if self.stack.layers.is_empty() {
                    let layer = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
                    self.stack.add(layer).unwrap();
                    self.thumb_textures.push(None);
                    self.thumb_dirty.push(true);
                }
                self.clamp_active();
                self.dirty  = true;
                self.marked.clear();
                self.status = "Layer deleted".into();
            }
            let n           = self.stack.layers.len();
            let can_move_up = self.active + 1 < n;
            let can_move_dn = self.active > 0;
            if ui.add_enabled(can_move_up, egui::Button::new("Up"))
                .on_hover_text("Move layer up (toward top)")
                .clicked()
            {
                self.stack.checkpoint();
                self.stack.reorder(self.active, self.active + 1);
                self.thumb_textures.swap(self.active, self.active + 1);
                self.thumb_dirty.swap(self.active, self.active + 1);
                self.active += 1;
                self.dirty   = true;
                self.marked.clear();
                self.status  = "Moved layer up".into();
            }
            if ui.add_enabled(can_move_dn, egui::Button::new("Dn"))
                .on_hover_text("Move layer down (toward bottom)")
                .clicked()
            {
                self.stack.checkpoint();
                self.stack.reorder(self.active, self.active - 1);
                self.thumb_textures.swap(self.active, self.active - 1);
                self.thumb_dirty.swap(self.active, self.active - 1);
                self.active -= 1;
                self.dirty   = true;
                self.marked.clear();
                self.status  = "Moved layer down".into();
            }
            if ui.button("Flatten").on_hover_text("Flatten visible").clicked() {
                self.stack.flatten_visible();
                self.active = 0;
                self.dirty  = true;
                self.thumb_invalidate_all();
                self.marked.clear();
                self.status = "Flattened".into();
            }
        });

        // Mark All / Clear Marks / Merge
        ui.horizontal(|ui| {
            if ui.small_button("Mark All").clicked() {
                self.marked = (0..self.stack.layers.len()).collect();
            }
            if ui.small_button("Clear Marks").clicked() {
                self.marked.clear();
            }
            let can_merge = self.marked.len() >= 2;
            if ui.add_enabled(can_merge, egui::Button::new("Merge"))
                .on_hover_text("Composite marked layers (in stack order) into one 'merged' layer")
                .clicked()
            {
                // Sort ascending = bottom-first in composite order.
                // self.marked stores STACK INDICES (same coordinate system as self.stack.layers[i]).
                // Non-contiguous marks are pulled together: composited in stack order,
                // result inserted at topmost-marked's adjusted position, scattered
                // marked layers removed; unmarked layers keep their relative order.
                let mut sorted: Vec<usize> = self.marked.iter().copied().collect();
                sorted.sort_unstable();

                eprintln!("[merge] marks={:?} stack_len={}", sorted, self.stack.layers.len());

                let layers_to_merge: Vec<Layer> = sorted.iter()
                    .map(|&i| self.stack.layers[i].clone())
                    .collect();
                let mut merged = composite(&layers_to_merge);
                merged.mode    = BlendMode::Normal;
                merged.opacity = 1.0;
                merged.visible = true;
                merged.name    = "merged".into();

                // After removing all `sorted.len()` marked layers in descending order,
                // the topmost's original index shifts left by (sorted.len()-1) because
                // that many lower-indexed removals precede it.
                let topmost   = *sorted.last().unwrap();
                let insert_at = topmost - (sorted.len() - 1);
                let n_merged  = sorted.len();

                self.stack.checkpoint();
                for &idx in sorted.iter().rev() {
                    self.stack.layers.remove(idx);
                    self.thumb_remove(idx);
                }
                self.stack.layers.insert(insert_at, merged);
                self.thumb_insert(insert_at);

                self.active = insert_at;
                self.dirty  = true;
                self.marked.clear();
                self.status = format!("Merged {n_merged} layers");
            }
            let n = self.marked.len();
            if n > 0 {
                ui.label(format!("{n} marked"));
            }
        });

        // ---- Quad mirror-tile row ----
        ui.horizontal(|ui| {
            let at_cap = self.stack.layers.len() >= MAX_LAYERS;
            ui.label("Tiles:");
            let tiles_label = format!("{}×{}", self.quad_tiles, self.quad_tiles);
            egui::ComboBox::from_id_salt("quad_tiles")
                .selected_text(tiles_label)
                .width(60.0)
                .show_ui(ui, |ui| {
                    for &t in &[2u32, 4, 8, 16, 32, 64] {
                        let lbl = format!("{t}×{t}");
                        ui.selectable_value(&mut self.quad_tiles, t, lbl);
                    }
                });
            if ui.add_enabled(!at_cap, egui::Button::new("Quad"))
                .on_hover_text("Mirror-tile active layer into N×N grid on a new layer above it")
                .clicked()
            {
                if at_cap {
                    self.status = format!("Layer cap ({MAX_LAYERS}) reached — cannot Quad");
                } else {
                    let n        = self.quad_tiles as usize;
                    let src_name = self.active_name();
                    let src      = &self.stack.layers[self.active];
                    let sw       = src.width  as usize;
                    let sh       = src.height as usize;
                    let new_rgba = quad_fn(&src.rgba, sw, sh, n);
                    let mut new_layer    = Layer::new(src.width, src.height, [0.0; 4]);
                    new_layer.rgba       = new_rgba;
                    new_layer.name       = format!("Quad {n}×{n}");
                    let insert_at        = self.active + 1;
                    self.stack.checkpoint();
                    self.stack.layers.insert(insert_at, new_layer);
                    self.thumb_insert(insert_at);
                    self.active = insert_at;
                    self.dirty  = true;
                    self.marked.clear();
                    self.status = format!("Quad {n}×{n} from \"{src_name}\"");
                }
            }
        });

        ui.separator();

        // ---- Layer list (top of stack = top of list) ----
        let n = self.stack.layers.len();
        egui::ScrollArea::vertical()
            .id_salt("layer_list")
            .max_height(ui.available_height() * 0.45)
            .show(ui, |ui| {
                for display_i in (0..n).rev() {
                    let is_active = display_i == self.active;

                    // Pre-collect read-only state (avoids borrow conflicts in the Frame closure)
                    let (label, visible, is_marked, thumb_id, opacity) = {
                        let l = &self.stack.layers[display_i];
                        let lbl = if l.name.is_empty() { format!("Layer {display_i}") } else { l.name.clone() };
                        let tid = self.thumb_textures
                            .get(display_i)
                            .and_then(|o| o.as_ref())
                            .map(|t| t.id());
                        (lbl, l.visible, self.marked.contains(&display_i), tid, l.opacity)
                    };

                    // Active row highlight; marked rows get a secondary tint
                    let bg = if is_active {
                        egui::Color32::from_rgb(50, 75, 115)
                    } else if is_marked {
                        egui::Color32::from_rgb(70, 55, 90)
                    } else {
                        egui::Color32::TRANSPARENT
                    };

                    let resp = egui::Frame::none()
                        .fill(bg)
                        .inner_margin(egui::Margin::symmetric(4.0, 2.0))
                        .show(ui, |ui| {
                            let mut new_marked    = is_marked;
                            let mut eye_clicked   = false;
                            let mut label_clicked = false;
                            let mut new_opacity   = opacity;

                            ui.horizontal(|ui| {
                                // Checkbox (mark)
                                ui.checkbox(&mut new_marked, "");

                                // Thumbnail (or placeholder while generating)
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(THUMB_PX, THUMB_PX),
                                    egui::Sense::hover(),
                                );
                                if let Some(tid) = thumb_id {
                                    ui.painter().image(
                                        tid,
                                        rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );
                                } else {
                                    ui.painter().rect_filled(rect, 2.0, egui::Color32::from_gray(45));
                                }

                                // Eye toggle
                                let eye = if visible { "Vis" } else { "Hid" };
                                if ui.small_button(eye).clicked() { eye_clicked = true; }

                                // Per-row opacity — drag left/right, or double-click to type
                                ui.add(
                                    egui::DragValue::new(&mut new_opacity)
                                        .range(0.0_f32..=1.0_f32)
                                        .speed(0.005)
                                        .fixed_decimals(2)
                                ).on_hover_text("Opacity (drag or double-click to type)");

                                // Layer name / select
                                if ui.selectable_label(is_active, &label).clicked() {
                                    label_clicked = true;
                                }
                            });

                            (new_marked, eye_clicked, label_clicked, new_opacity)
                        });

                    // Apply interactions (after the closure so no borrow conflicts).
                    // display_i == stack index (loop iterates n-1..=0, same as stack index).
                    if resp.inner.0 != is_marked {
                        if resp.inner.0 {
                            eprintln!("[mark] checked stack[{display_i}]  marks={:?}",
                                      { let mut m = self.marked.clone(); m.insert(display_i); m });
                            self.marked.insert(display_i);
                        } else {
                            eprintln!("[mark] unchecked stack[{display_i}]");
                            self.marked.remove(&display_i);
                        }
                    }
                    if resp.inner.1 {
                        self.stack.layers[display_i].visible ^= true;
                        self.dirty = true;
                    }
                    if resp.inner.2 {
                        self.active = display_i;
                    }
                    let new_op = resp.inner.3;
                    if (new_op - opacity).abs() > 1e-4 {
                        self.stack.layers[display_i].opacity = new_op;
                        self.dirty = true;
                    }
                }
            });

        // ---- Active layer detail (mode, opacity) ----
        if let Some(layer) = self.stack.layers.get_mut(self.active) {
            ui.separator();
            ui.label(format!("Active: {}",
                if layer.name.is_empty() { format!("Layer {}", self.active) } else { layer.name.clone() }
            ));
            ui.horizontal(|ui| {
                ui.label("Mode:");
                egui::ComboBox::from_id_salt("blend_mode")
                    .selected_text(mode_label(layer.mode))
                    .width(110.0)
                    .show_ui(ui, |ui| {
                        for &mode in MODES {
                            if ui.selectable_label(layer.mode == mode, mode_label(mode)).clicked() {
                                layer.mode = mode;
                                self.dirty = true;
                            }
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Opacity:");
                let mut op = layer.opacity;
                if ui.add(egui::Slider::new(&mut op, 0.0..=1.0).fixed_decimals(2)).changed() {
                    layer.opacity = op;
                    self.dirty    = true;
                }
            });
        }

        // ---- Select by Color ----
        ui.separator();
        ui.heading("Select by Color");
        ui.horizontal(|ui| {
            ui.label("Aggr %:");
            ui.add(egui::Slider::new(&mut self.selection_aggr, 0.0_f32..=100.0).fixed_decimals(0));
        });
        ui.horizontal(|ui| {
            ui.label("Feather %:");
            ui.add(egui::Slider::new(&mut self.selection_feather, 0.0_f32..=100.0).fixed_decimals(0));
        });
        {
            // sqrt(3) ≈ 1.732 is the max Euclidean distance in the unit RGB cube.
            // Apply the same power curve used at pick time so the label matches reality.
            let max_d = 3.0_f32.sqrt();
            let tol = (self.selection_aggr   / 100.0).powf(AGGR_GAMMA)    * max_d;
            let fth = (self.selection_feather / 100.0).powf(FEATHER_GAMMA) * max_d;
            ui.label(
                egui::RichText::new(format!(
                    "tol {:.2} / feather {:.2}  (max {:.2})", tol, fth, max_d,
                ))
                .small()
                .weak(),
            );
        }
        ui.horizontal(|ui| {
            let pick_lbl = if self.selection_pick_mode { "◉ Pick (click canvas)" } else { "Pick" };
            if ui.selectable_label(self.selection_pick_mode, pick_lbl).clicked() {
                self.selection_pick_mode ^= true;
            }
            if ui.button("Clear Selection").clicked() {
                self.clear_selection();
                self.selection_pick_mode = false;
                self.status              = "Selection cleared".into();
            }
        });
        if self.stack.has_selection {
            ui.label(format!("Selected: {:.1}%", self.selection_coverage_pct));
        }

        // ---- Adjustments ----
        // Template: each adjustment calls adjust::apply_adjustment through the mask.
        // To add a new one: write a sibling fn in adjust.rs + a button here.
        ui.separator();
        ui.heading("Adjustments");
        if ui.button("Invert")
            .on_hover_text("Perceptual (sRGB) invert — whole layer or clipped to selection")
            .clicked()
        {
            if self.stack.layers.is_empty() {
                self.status = "Invert: no active layer".into();
            } else {
                let idx  = self.active;
                let name = self.active_name();
                let lw   = self.stack.layers[idx].width  as usize;
                let lh   = self.stack.layers[idx].height as usize;
                let n_px = lw * lh;

                let mask_matches = self.stack.has_selection && self.stack.selection_mask.len() == n_px;
                // Clone the mask before the mutable borrow in snapshot_layer; the
                // immutable borrow of self.stack.selection_mask must not outlive it.
                let mask_clone: Option<Vec<f32>> = if mask_matches {
                    Some(self.stack.selection_mask.clone())
                } else {
                    None
                };

                self.stack.snapshot_layer(idx);
                adjust_invert(&mut self.stack.layers[idx].rgba, mask_clone.as_deref());
                self.dirty = true;
                self.thumb_invalidate(idx);

                self.status = if mask_matches {
                    format!("Inverted selection on {name}")
                } else if self.stack.has_selection {
                    // mask existed but dims didn't match — fell back to whole layer
                    format!("Inverted {name} (selection size mismatch — whole layer)")
                } else {
                    format!("Inverted {name}")
                };
            }
        }

        // ---- Plasma ----
        ui.separator();
        ui.heading("Plasma");
        ui.horizontal(|ui| {
            ui.label("Seed:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.plasma_seed_str).desired_width(90.0));
            if resp.lost_focus() || resp.changed() {
                if let Ok(v) = self.plasma_seed_str.parse::<u64>() { self.plasma_seed = v; }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Turb:");
            ui.add(egui::Slider::new(&mut self.plasma_turbulence, 0.3..=2.5).fixed_decimals(2));
        });
        ui.horizontal(|ui| {
            if ui.button("New Seed").clicked() {
                self.plasma_seed     = time_seed();
                self.plasma_seed_str = self.plasma_seed.to_string();
            }
            let in_flight = self.render_rx.is_some();
            if ui.add_enabled(!in_flight, egui::Button::new("Apply")).clicked() {
                let seed = self.plasma_seed;
                let turb = self.plasma_turbulence as f64;
                let idx  = self.active;
                let name = self.active_name();
                self.stack.checkpoint();
                let progress = Arc::new(AtomicU32::new(0));
                self.render_progress   = Arc::clone(&progress);
                self.render_total_rows = IMG_H;
                self.render_target_idx = idx;
                self.render_finish     = format!("Plasma on {name} (seed={seed})");
                self.render_label      = "Plasma".into();
                let (tx, rx) = mpsc::channel();
                self.render_rx = Some(rx);
                std::thread::spawn(move || {
                    let mut buf = vec![0.0f32; IMG_W * IMG_H * 4];
                    apply_plasma_with_progress(&mut buf, seed, turb, &progress);
                    let _ = tx.send(RenderBuf::Linear(buf));
                });
            }
        });
        if self.render_rx.is_some() && self.render_label == "Plasma" {
            let done  = self.render_progress.load(Ordering::Relaxed) as f32;
            let total = self.render_total_rows as f32;
            let frac  = if total > 0.0 { (done / total).min(1.0) } else { 0.0 };
            let pct   = (frac * 100.0) as u32;
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("Rendering plasma… {pct}%"))
                    .animate(true),
            );
        }

        // ---- Qbist ----
        ui.separator();
        ui.heading("Qbist");
        ui.horizontal(|ui| {
            ui.label("Seed:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.qbist_seed_str).desired_width(90.0),
            );
            if resp.lost_focus() || resp.changed() {
                // Accept decimal or 0x-prefixed hex
                let parsed = if let Some(h) = self.qbist_seed_str
                    .strip_prefix("0x")
                    .or_else(|| self.qbist_seed_str.strip_prefix("0X"))
                {
                    u64::from_str_radix(h, 16).ok()
                } else {
                    self.qbist_seed_str.parse::<u64>().ok()
                };
                if let Some(v) = parsed {
                    self.qbist_seed = v;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Oversampling:");
            let mut os = self.qbist_oversampling as i32;
            if ui.add(egui::DragValue::new(&mut os).range(1..=4)).changed() {
                self.qbist_oversampling = os as u8;
            }
        });
        ui.horizontal(|ui| {
            if ui.button("New Seed").clicked() {
                self.qbist_seed     = time_seed();
                self.qbist_seed_str = self.qbist_seed.to_string();
            }
            let in_flight = self.render_rx.is_some();
            if ui.add_enabled(!in_flight, egui::Button::new("Apply")).clicked() {
                let seed = self.qbist_seed;
                let os   = self.qbist_oversampling as usize;
                let idx  = self.active;
                let w    = self.stack.layers[idx].width  as usize;
                let h    = self.stack.layers[idx].height as usize;
                let name = self.active_name();
                let mut genome = create_info(seed);
                let (used_trans, used_reg) = optimize(&mut genome);
                self.stack.checkpoint();
                let progress = Arc::new(AtomicU32::new(0));
                self.render_progress   = Arc::clone(&progress);
                self.render_total_rows = h;
                self.render_target_idx = idx;
                self.render_finish     = format!("Qbist on {name} (seed={seed} os={os})");
                self.render_label      = "Qbist".into();
                let (tx, rx) = mpsc::channel();
                self.render_rx = Some(rx);
                std::thread::spawn(move || {
                    let buf = qbist_render_threaded(
                        &genome, &used_trans, &used_reg, w, h, os, &progress,
                    );
                    let _ = tx.send(RenderBuf::SrgbU8(buf));
                });
            }
        });
        if self.render_rx.is_some() && self.render_label == "Qbist" {
            let done  = self.render_progress.load(Ordering::Relaxed) as f32;
            let total = self.render_total_rows as f32;
            let frac  = if total > 0.0 { (done / total).min(1.0) } else { 0.0 };
            let pct   = (frac * 100.0) as u32;
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("Rendering qbist… {pct}%"))
                    .animate(true),
            );
        }

        // ---- Spiral ----
        ui.separator();
        ui.heading("Spiral");
        ui.horizontal(|ui| {
            ui.label("Type:");
            ui.radio_value(&mut self.spiral_kind, SpiralKind::Logarithmic, "Logarithmic");
            ui.radio_value(&mut self.spiral_kind, SpiralKind::Archimedean, "Archimedean");
        });
        ui.horizontal(|ui| {
            ui.label("Turns:");
            let mut t_i = self.spiral_turns as i32;
            if ui.add(egui::Slider::new(&mut t_i, 2..=60)).changed() {
                self.spiral_turns = t_i as u32;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Arms:");
            let mut arms_i = self.spiral_arms as i32;
            if ui.add(egui::Slider::new(&mut arms_i, 1..=12)).changed() {
                self.spiral_arms = arms_i as u32;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Color A:");
            egui::color_picker::color_edit_button_srgb(ui, &mut self.spiral_color_a);
            ui.label("Color B:");
            egui::color_picker::color_edit_button_srgb(ui, &mut self.spiral_color_b);
        });
        {
            let in_flight = self.render_rx.is_some();
            if ui.add_enabled(!in_flight, egui::Button::new("Apply Spiral")).clicked() {
                let kind  = self.spiral_kind;
                let turns = self.spiral_turns;
                let arms  = self.spiral_arms;
                let ca = {
                    let [r, g, b] = self.spiral_color_a;
                    [
                        srgb_to_linear(r as f32 / 255.0),
                        srgb_to_linear(g as f32 / 255.0),
                        srgb_to_linear(b as f32 / 255.0),
                        1.0,
                    ]
                };
                let cb = {
                    let [r, g, b] = self.spiral_color_b;
                    [
                        srgb_to_linear(r as f32 / 255.0),
                        srgb_to_linear(g as f32 / 255.0),
                        srgb_to_linear(b as f32 / 255.0),
                        1.0,
                    ]
                };
                let idx  = self.active;
                let w    = self.stack.layers[idx].width  as usize;
                let h    = self.stack.layers[idx].height as usize;
                let name = self.active_name();
                self.stack.checkpoint();
                let progress = Arc::new(AtomicU32::new(0));
                self.render_progress   = Arc::clone(&progress);
                self.render_total_rows = h;
                self.render_target_idx = idx;
                self.render_finish     = format!("Spiral on {name}");
                self.render_label      = "Spiral".into();
                let (tx, rx) = mpsc::channel();
                self.render_rx = Some(rx);
                std::thread::spawn(move || {
                    let mut buf = vec![0.0f32; w * h * 4];
                    spiral_render_threaded(
                        &mut buf, w, h, ca, cb, kind, turns, arms, SPIRAL_OS, &progress,
                    );
                    let _ = tx.send(RenderBuf::Linear(buf));
                });
            }
        }
        if self.render_rx.is_some() && self.render_label == "Spiral" {
            let done  = self.render_progress.load(Ordering::Relaxed) as f32;
            let total = self.render_total_rows as f32;
            let frac  = if total > 0.0 { (done / total).min(1.0) } else { 0.0 };
            let pct   = (frac * 100.0) as u32;
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("Rendering spiral… {pct}%"))
                    .animate(true),
            );
        }

        // ---- Punch ----
        ui.separator();
        ui.heading("Punch");
        ui.horizontal(|ui| {
            ui.label("Contrast:");
            ui.add(egui::Slider::new(&mut self.punch_contrast, 0.0..=9.0).fixed_decimals(1));
        });
        ui.horizontal(|ui| {
            ui.label("Saturation:");
            ui.add(egui::Slider::new(&mut self.punch_saturation, 0.0..=4.0).fixed_decimals(1));
        });
        ui.horizontal(|ui| {
            ui.label("Passes:");
            let mut p = self.punch_passes as i32;
            if ui.add(egui::Slider::new(&mut p, 1..=6)).changed() { self.punch_passes = p as u32; }
        });
        if ui.button("Apply Punch").clicked() {
            let k    = self.punch_contrast;
            let sat  = self.punch_saturation;
            let pass = self.punch_passes;
            let idx  = self.active;
            let lw   = self.stack.layers[idx].width  as usize;
            let lh   = self.stack.layers[idx].height as usize;
            let n_px = lw * lh;
            let use_mask = self.stack.has_selection && self.stack.selection_mask.len() == n_px;
            self.stack.snapshot_layer(idx);
            // Clone pre-punch pixels for masked blend (only when selection is active).
            let pre_punch = if use_mask {
                Some(self.stack.layers[idx].rgba.clone())
            } else {
                None
            };
            punch(&mut self.stack.layers[idx].rgba, k, sat, pass);
            if let Some(old) = pre_punch {
                for i in 0..n_px {
                    let m = self.stack.selection_mask[i];
                    let s = i * 4;
                    for c in 0..4usize {
                        let punched = self.stack.layers[idx].rgba[s + c];
                        self.stack.layers[idx].rgba[s + c] =
                            old[s + c] * (1.0 - m) + punched * m;
                    }
                }
            }
            self.dirty = true;
            self.thumb_invalidate(idx);
            let name = self.active_name();
            self.status = format!("Punch on {name} (k={k:.1} sat={sat:.1} ×{pass})");
        }

        // ---- Edge ----
        ui.separator();
        ui.heading("Edge");
        ui.horizontal(|ui| {
            ui.label("Aggr %:");
            let n = aggr_to_n(self.edge_aggr);
            ui.add(egui::Slider::new(&mut self.edge_aggr, 0.0..=100.0).fixed_decimals(0))
                .on_hover_text(format!("N={n} levels"));
        });
        ui.horizontal(|ui| {
            ui.label("Width px:");
            let mut w = self.edge_width as i32;
            if ui.add(egui::Slider::new(&mut w, 1..=7)).changed() { self.edge_width = w as usize; }
        });
        if ui.button("Apply Edge").clicked() && !self.stack.layers.is_empty() {
            let aggr  = self.edge_aggr;
            let width = self.edge_width;
            let idx   = self.active;
            self.stack.checkpoint();
            let src      = &self.stack.layers[idx];
            let (pw, ph) = (src.width as usize, src.height as usize);
            let edge_buf = edge(&src.rgba.clone(), pw, ph, aggr, width);
            let mut edge_layer    = Layer::new(pw as u32, ph as u32, [0.0; 4]);
            edge_layer.rgba       = edge_buf;
            edge_layer.name       = "edge".into();
            self.stack.layers.insert(idx + 1, edge_layer);
            self.thumb_insert(idx + 1);
            self.marked.clear();
            self.active = idx + 1;
            self.dirty  = true;
            let n = aggr_to_n(aggr);
            self.status = format!("Edge layer created (N={n} w={width})");
        }

        // ---- Band Generator ----
        ui.separator();
        ui.heading("Band Generator");
        ui.horizontal(|ui| {
            let t = growth_to_thickness(self.band_growth);
            ui.label("Growth %:");
            ui.add(egui::Slider::new(&mut self.band_growth, 0.0..=100.0).fixed_decimals(0))
                .on_hover_text(format!("thickness = {t} px"));
        });
        ui.horizontal(|ui| {
            ui.label("Fill %:");
            ui.add(egui::Slider::new(&mut self.band_fill, 0.0..=100.0).fixed_decimals(0));
        });
        if ui.button("Generate").clicked() {
            self.run_band_generate();
        }

        // ---- Kaleidoscope ----
        ui.separator();
        ui.heading("Kaleidoscope");
        ui.horizontal(|ui| {
            ui.label("Segments:");
            let mut s = self.kaleido_segments;
            if ui.add(egui::Slider::new(&mut s, 2..=24)).changed() {
                self.kaleido_segments = s;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Rotation:");
            ui.add(egui::Slider::new(&mut self.kaleido_rotation, 0.0..=360.0).fixed_decimals(0));
        });
        ui.horizontal(|ui| {
            ui.label("Zoom:");
            ui.add(egui::Slider::new(&mut self.kaleido_zoom, 0.25..=4.0).fixed_decimals(2));
        });
        let at_cap = self.stack.layers.len() >= MAX_LAYERS;
        if ui.add_enabled(!at_cap, egui::Button::new("Apply Kaleidoscope"))
            .on_hover_text("Kaleidoscope each marked layer independently → one new output per source appended above stack")
            .clicked()
        {
            self.run_kaleido();
        }

        if at_cap {
            ui.colored_label(egui::Color32::YELLOW, format!("Layer cap: {MAX_LAYERS}"));
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("basecoat")
            .with_inner_size([1400.0, 860.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "basecoat",
        opts,
        Box::new(|_cc| Ok(Box::new(BasecoatApp::new()))),
    )
}

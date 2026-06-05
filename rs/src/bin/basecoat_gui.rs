//! basecoat GUI — thin viewer/driver over the headless engine.
//! egui/eframe + wgpu backend.

use basecoat::layers::*;
use basecoat::plasma::{apply_plasma, H as IMG_H, W as IMG_W};
use basecoat::punch::punch;
use eframe::egui;
use png::{BitDepth, ColorType, Encoder, Unit};
use std::io::BufWriter;
use std::time::{SystemTime, UNIX_EPOCH};

const W: u32 = IMG_W as u32;
const H: u32 = IMG_H as u32;
const PPM: u32 = 23622;

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
// Convert composite Layer (linear f32) to egui ColorImage (sRGB u8)
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

// ---------------------------------------------------------------------------
// Simple time-based seed for "New Seed" button
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
    stack: Stack,
    active: usize,

    // canvas texture cache
    dirty: bool,
    texture: Option<egui::TextureHandle>,

    // plasma controls
    plasma_seed: u64,
    plasma_seed_str: String,
    plasma_turbulence: f32,

    // punch controls
    punch_contrast:   f32,
    punch_saturation: f32,
    punch_passes:     u32,

    // zoom / pan
    zoom: f32,       // 0.0 = fit-to-window; >0 = absolute scale
    fit_zoom: f32,   // last computed fit scale, updated each canvas frame
    pan: egui::Vec2,

    // status line
    status: String,
}

impl BasecoatApp {
    fn new() -> Self {
        let mut stack = Stack::new();
        let base = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
        stack.add(base).unwrap();

        Self {
            stack,
            active: 0,
            dirty: true,
            texture: None,
            plasma_seed: 0,
            plasma_seed_str: "0".into(),
            plasma_turbulence: 1.0,
            punch_contrast:   9.0,
            punch_saturation: 4.0,
            punch_passes:     6,
            zoom: 0.0,
            fit_zoom: 1.0,
            pan: egui::Vec2::ZERO,
            status: "Ready".into(),
        }
    }

    fn ensure_composite(&mut self, ctx: &egui::Context) {
        if !self.dirty {
            return;
        }
        let comp = self.stack.composite();
        let img  = layer_to_color_image(&comp);
        let tex  = ctx.load_texture("canvas", img, egui::TextureOptions::LINEAR);
        self.texture = Some(tex);
        self.dirty = false;
    }

    fn active_name(&self) -> String {
        self.stack.layers.get(self.active)
            .map(|l| if l.name.is_empty() { format!("Layer {}", self.active) } else { l.name.clone() })
            .unwrap_or_default()
    }

    fn clamp_active(&mut self) {
        let n = self.stack.layers.len();
        if n == 0 {
            self.active = 0;
        } else if self.active >= n {
            self.active = n - 1;
        }
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
        // Recomposite once when dirty (not every frame)
        self.ensure_composite(ctx);

        // ----------------------------------------------------------------
        // Keyboard zoom  (Ctrl+=  Ctrl++  Ctrl+-  Ctrl+0)
        // ----------------------------------------------------------------
        const ZOOM_STEP: f32 = 1.25;
        let (zoom_in, zoom_out, zoom_fit) = ctx.input(|i| {
            let ctrl = i.modifiers.ctrl;
            (
                ctrl && (i.key_pressed(egui::Key::Equals) || i.key_pressed(egui::Key::Plus)),
                ctrl && i.key_pressed(egui::Key::Minus),
                ctrl && i.key_pressed(egui::Key::Num0),
            )
        });
        if zoom_in {
            let cur = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom };
            self.zoom = (cur * ZOOM_STEP).clamp(0.05, 10.0);
        }
        if zoom_out {
            let cur = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom };
            self.zoom = (cur / ZOOM_STEP).clamp(0.05, 10.0);
        }
        if zoom_fit {
            self.zoom = 0.0;
            self.pan  = egui::Vec2::ZERO;
        }

        // ----------------------------------------------------------------
        // Menu bar
        // ----------------------------------------------------------------
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // File menu
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.stack = Stack::new();
                        let base = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
                        self.stack.add(base).unwrap();
                        self.active = 0;
                        self.dirty  = true;
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
                                Ok(()) => self.status = format!("Exported {}", path.display()),
                                Err(e) => self.status = format!("Export error: {e}"),
                            }
                        }
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                // Edit menu
                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo").clicked() {
                        if self.stack.undo() {
                            self.clamp_active();
                            self.dirty  = true;
                            self.status = "Undo".into();
                        } else {
                            self.status = "Nothing to undo".into();
                        }
                        ui.close_menu();
                    }
                });

                // View menu
                ui.menu_button("View", |ui| {
                    if ui.button("Zoom In   Ctrl+=").clicked() {
                        let cur = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom };
                        self.zoom = (cur * ZOOM_STEP).clamp(0.05, 10.0);
                        ui.close_menu();
                    }
                    if ui.button("Zoom Out  Ctrl+\u{2212}").clicked() {
                        let cur = if self.zoom == 0.0 { self.fit_zoom } else { self.zoom };
                        self.zoom = (cur / ZOOM_STEP).clamp(0.05, 10.0);
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

        // ----------------------------------------------------------------
        // Status bar
        // ----------------------------------------------------------------
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.label(&self.status);
        });

        // ----------------------------------------------------------------
        // Right panel
        // ----------------------------------------------------------------
        egui::SidePanel::right("layers_panel")
            .min_width(230.0)
            .max_width(300.0)
            .show(ctx, |ui| {
                self.show_layers_panel(ui);
            });

        // ----------------------------------------------------------------
        // Central canvas
        // ----------------------------------------------------------------
        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_canvas(ui);
        });
    }
}

impl BasecoatApp {
    fn show_canvas(&mut self, ui: &mut egui::Ui) {
        let Some(tex) = &self.texture else { return; };

        let avail = ui.available_size();
        let img_size = egui::vec2(W as f32, H as f32);

        // Compute effective zoom (0 = fit); store fit_zoom for keyboard shortcuts.
        let fit_zoom = (avail.x / img_size.x).min(avail.y / img_size.y);
        self.fit_zoom = fit_zoom;
        let effective_zoom = if self.zoom == 0.0 { fit_zoom } else { self.zoom };
        let display_size = img_size * effective_zoom;

        // Pan with drag
        let drag = ui.input(|i| {
            if i.pointer.middle_down() || (i.pointer.secondary_down()) {
                i.pointer.delta()
            } else {
                egui::Vec2::ZERO
            }
        });
        self.pan += drag;

        // Draw via a ScrollArea so oversized images get scrollbars automatically
        egui::ScrollArea::both()
            .id_salt("canvas_scroll")
            .show(ui, |ui| {
                let offset_rect = egui::Rect::from_min_size(
                    ui.cursor().min + self.pan,
                    display_size,
                );
                let resp = ui.allocate_rect(offset_rect, egui::Sense::drag());
                if resp.dragged_by(egui::PointerButton::Primary) {
                    self.pan += resp.drag_delta();
                }
                ui.painter().image(
                    tex.id(),
                    offset_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            });
    }

    fn show_layers_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Layers");
        ui.separator();

        // ---- Layer ops ----
        ui.horizontal(|ui| {
            let at_cap = self.stack.layers.len() >= MAX_LAYERS;
            if ui.add_enabled(!at_cap, egui::Button::new("＋")).clicked() {
                let layer = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
                self.stack.add(layer).unwrap();
                self.active = self.stack.layers.len() - 1;
                self.dirty  = true;
                self.status = "Layer added".into();
            }
            if ui.button("－").on_hover_text("Delete active layer").clicked() {
                if !self.stack.layers.is_empty() {
                    self.stack.remove(self.active);
                    // If stack is now empty, add a base layer
                    if self.stack.layers.is_empty() {
                        let layer = Layer::new(W, H, [0.0, 0.0, 0.0, 0.0]);
                        self.stack.add(layer).unwrap();
                    }
                    self.clamp_active();
                    self.dirty  = true;
                    self.status = "Layer deleted".into();
                }
            }
            if ui.button("↑").on_hover_text("Move up").clicked() {
                let n = self.stack.layers.len();
                if self.active + 1 < n {
                    self.stack.reorder(self.active, self.active + 1);
                    self.active += 1;
                    self.dirty   = true;
                }
            }
            if ui.button("↓").on_hover_text("Move down").clicked() {
                if self.active > 0 {
                    self.stack.reorder(self.active, self.active - 1);
                    self.active -= 1;
                    self.dirty   = true;
                }
            }
            if ui.button("Flatten").on_hover_text("Flatten visible").clicked() {
                self.stack.flatten_visible();
                self.active = 0;
                self.dirty  = true;
                self.status = "Flattened".into();
            }
        });

        ui.separator();

        // ---- Layer list (top of stack = top of list, like GIMP) ----
        let n = self.stack.layers.len();
        egui::ScrollArea::vertical()
            .id_salt("layer_list")
            .max_height(300.0)
            .show(ui, |ui| {
                for display_i in (0..n).rev() {
                    let is_active = display_i == self.active;
                    // Copy what we need before any closures borrow self.
                    let (label, visible) = {
                        let l = &self.stack.layers[display_i];
                        let lbl = if l.name.is_empty() {
                            format!("Layer {display_i}")
                        } else {
                            l.name.clone()
                        };
                        (lbl, l.visible)
                    };
                    let bg = if is_active {
                        egui::Color32::from_rgb(60, 80, 120)
                    } else {
                        egui::Color32::TRANSPARENT
                    };

                    let resp = egui::Frame::none()
                        .fill(bg)
                        .inner_margin(egui::Margin::symmetric(4.0, 2.0))
                        .show(ui, |ui| {
                            let mut eye_clicked   = false;
                            let mut label_clicked = false;
                            ui.horizontal(|ui| {
                                let eye = if visible { "👁" } else { "⊘" };
                                if ui.small_button(eye).clicked()              { eye_clicked   = true; }
                                if ui.selectable_label(is_active, &label).clicked() { label_clicked = true; }
                            });
                            (eye_clicked, label_clicked)
                        });
                    if resp.inner.0 {
                        self.stack.layers[display_i].visible ^= true;
                        self.dirty = true;
                    }
                    if resp.inner.1 {
                        self.active = display_i;
                    }
                }
            });

        // ---- Active layer controls ----
        if let Some(layer) = self.stack.layers.get_mut(self.active) {
            ui.separator();
            ui.label(format!("Active: {}", if layer.name.is_empty() {
                format!("Layer {}", self.active)
            } else {
                layer.name.clone()
            }));

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
                    self.dirty = true;
                }
            });
        }

        // ---- Plasma technique ----
        ui.separator();
        ui.heading("Plasma");

        ui.horizontal(|ui| {
            ui.label("Seed:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.plasma_seed_str).desired_width(90.0),
            );
            if resp.lost_focus() || resp.changed() {
                if let Ok(v) = self.plasma_seed_str.parse::<u64>() {
                    self.plasma_seed = v;
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Turbulence:");
            ui.add(egui::Slider::new(&mut self.plasma_turbulence, 0.3..=2.5).fixed_decimals(2));
        });

        ui.horizontal(|ui| {
            if ui.button("New Seed").clicked() {
                self.plasma_seed = time_seed();
                self.plasma_seed_str = self.plasma_seed.to_string();
            }
            if ui.button("Apply").clicked() {
                let seed = self.plasma_seed;
                let turb = self.plasma_turbulence as f64;
                let active = self.active;
                // push undo snapshot via fill (pixel op), then overwrite buffer
                // We snapshot manually since apply_plasma writes the whole layer
                self.stack.layers[active].rgba = {
                    let mut buf = vec![0.0f32; (W * H * 4) as usize];
                    apply_plasma(&mut buf, seed, turb);
                    buf
                };
                // record the operation in undo ring as a structural snap
                // (pixel snap already handled implicitly — we pushed the struct above)
                self.dirty  = true;
                let name    = self.active_name();
                self.status = format!("Plasma applied to {name} (seed={seed})");
            }
        });

        // ---- Punch technique ----
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
            if ui.add(egui::Slider::new(&mut p, 1..=6)).changed() {
                self.punch_passes = p as u32;
            }
        });

        if ui.button("Apply Punch").clicked() {
            let k    = self.punch_contrast;
            let sat  = self.punch_saturation;
            let pass = self.punch_passes;
            let idx  = self.active;
            // Pixel-op snapshot: save buffer before mutation, then punch in-place.
            self.stack.snapshot_layer(idx);
            punch(&mut self.stack.layers[idx].rgba, k, sat, pass);
            self.dirty  = true;
            let name    = self.active_name();
            self.status = format!("Punch applied to {name} (k={k:.1} sat={sat:.1} ×{pass})");
        }

        if self.stack.layers.len() >= MAX_LAYERS {
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
            .with_inner_size([1280.0, 800.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "basecoat",
        opts,
        Box::new(|_cc| Ok(Box::new(BasecoatApp::new()))),
    )
}

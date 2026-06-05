//! Layer stack + compositor engine — see spec/layers.md.

pub const MAX_LAYERS: usize = 25;
pub const UNDO_DEPTH: usize = 20;

// ---------------------------------------------------------------------------
// sRGB <-> linear  (IEC 61966-2-1)
// ---------------------------------------------------------------------------

pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode linear float to sRGB; runs in f64 to match Python's ndarray f64 path.
pub fn linear_to_srgb_f64(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ---------------------------------------------------------------------------
// Blend modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Difference,
}

/// Blend top channel `a` over bottom `b` — linear light, f64.
#[inline]
pub fn blend_channel(mode: BlendMode, a: f64, b: f64) -> f64 {
    match mode {
        BlendMode::Normal     => a,
        BlendMode::Multiply   => a * b,
        BlendMode::Screen     => 1.0 - (1.0 - a) * (1.0 - b),
        BlendMode::Overlay    => {
            if b < 0.5 { 2.0 * a * b } else { 1.0 - 2.0 * (1.0 - a) * (1.0 - b) }
        }
        BlendMode::Difference => (a - b).abs(),
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Layer {
    /// Linear-light straight-alpha RGBA, row-major, f32 per channel, [0,1].
    pub rgba: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub mode: BlendMode,
    pub opacity: f32,
    pub visible: bool,
    pub name: String,
}

impl Layer {
    pub fn new(width: u32, height: u32, color: [f32; 4]) -> Self {
        let n = (width * height * 4) as usize;
        let rgba = (0..n).map(|i| color[i % 4]).collect();
        Self {
            rgba,
            width,
            height,
            mode: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            name: String::new(),
        }
    }

    pub fn fill(&mut self, color: [f32; 4]) {
        for chunk in self.rgba.chunks_exact_mut(4) {
            chunk.copy_from_slice(&color);
        }
    }
}

// ---------------------------------------------------------------------------
// Composite  — accumulates in f64 throughout, stores result as f32.
// This matches Python's: acc in float64, .astype(np.float32) at the end.
// ---------------------------------------------------------------------------

pub fn composite(layers: &[Layer]) -> Layer {
    let first = layers.first().expect("empty layer list");
    let (w, h) = (first.width, first.height);
    let n = (w * h) as usize;

    // Four f64 channels: R G B A
    let mut acc_r = vec![0.0f64; n];
    let mut acc_g = vec![0.0f64; n];
    let mut acc_b = vec![0.0f64; n];
    let mut acc_a = vec![0.0f64; n];

    for layer in layers {
        if !layer.visible {
            continue;
        }
        let opacity = layer.opacity as f64;
        let mode    = layer.mode;

        for idx in 0..n {
            let i = idx * 4;
            let cs_r = layer.rgba[i    ] as f64;
            let cs_g = layer.rgba[i + 1] as f64;
            let cs_b = layer.rgba[i + 2] as f64;
            let cs_a = layer.rgba[i + 3] as f64;

            let as_ = cs_a * opacity;
            let ab  = acc_a[idx];
            let aout = as_ + ab * (1.0 - as_);

            if aout > 0.0 {
                let inv = 1.0 - as_;
                acc_r[idx] = (blend_channel(mode, cs_r, acc_r[idx]) * as_
                              + acc_r[idx] * ab * inv) / aout;
                acc_g[idx] = (blend_channel(mode, cs_g, acc_g[idx]) * as_
                              + acc_g[idx] * ab * inv) / aout;
                acc_b[idx] = (blend_channel(mode, cs_b, acc_b[idx]) * as_
                              + acc_b[idx] * ab * inv) / aout;
                acc_a[idx] = aout;
            }
            // if aout == 0: pixel stays zero (transparent black)
        }
    }

    // Clamp and convert to f32 — same as Python's np.clip(...).astype(np.float32)
    let mut rgba = vec![0.0f32; n * 4];
    for idx in 0..n {
        let i = idx * 4;
        rgba[i    ] = acc_r[idx].clamp(0.0, 1.0) as f32;
        rgba[i + 1] = acc_g[idx].clamp(0.0, 1.0) as f32;
        rgba[i + 2] = acc_b[idx].clamp(0.0, 1.0) as f32;
        rgba[i + 3] = acc_a[idx].clamp(0.0, 1.0) as f32;
    }

    Layer {
        rgba,
        width: w,
        height: h,
        mode: BlendMode::Normal,
        opacity: 1.0,
        visible: true,
        name: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Undo snapshots
// ---------------------------------------------------------------------------

enum Snap {
    Pixel  { idx: usize, buf: Vec<f32> },
    Struct { layers: Vec<Layer> },
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

pub struct Stack {
    pub layers: Vec<Layer>,
    undo: Vec<Snap>,
}

impl Stack {
    pub fn new() -> Self {
        Self { layers: Vec::new(), undo: Vec::new() }
    }

    /// Snapshot one layer's pixel buffer for undo without modifying it.
    /// Use before any destructive pixel op (punch, custom fill, etc.).
    pub fn snapshot_layer(&mut self, idx: usize) {
        self.push_pixel_snap(idx);
    }

    fn push_pixel_snap(&mut self, idx: usize) {
        let buf = self.layers[idx].rgba.clone();
        self.undo.push(Snap::Pixel { idx, buf });
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    fn push_struct_snap(&mut self) {
        self.undo.push(Snap::Struct { layers: self.layers.clone() });
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    pub fn add(&mut self, layer: Layer) -> Result<(), &'static str> {
        if self.layers.len() >= MAX_LAYERS {
            return Err("MAX_LAYERS reached");
        }
        self.push_struct_snap();
        self.layers.push(layer);
        Ok(())
    }

    pub fn remove(&mut self, idx: usize) {
        self.push_struct_snap();
        self.layers.remove(idx);
    }

    pub fn reorder(&mut self, from_idx: usize, to_idx: usize) {
        self.push_struct_snap();
        let layer = self.layers.remove(from_idx);
        self.layers.insert(to_idx, layer);
    }

    pub fn flatten_visible(&mut self) {
        self.push_struct_snap();
        let visible: Vec<usize> = self.layers.iter().enumerate()
            .filter(|(_, l)| l.visible)
            .map(|(i, _)| i)
            .collect();
        if visible.is_empty() {
            return;
        }
        let mut merged = composite(&self.layers);
        merged.mode    = BlendMode::Normal;
        merged.opacity = 1.0;
        merged.visible = true;
        merged.name    = "merged".into();
        let insert_at = visible[0];
        for &i in visible.iter().rev() {
            self.layers.remove(i);
        }
        self.layers.insert(insert_at, merged);
    }

    pub fn fill(&mut self, idx: usize, color: [f32; 4]) {
        self.push_pixel_snap(idx);
        self.layers[idx].fill(color);
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            None => false,
            Some(Snap::Pixel { idx, buf }) => {
                self.layers[idx].rgba = buf;
                true
            }
            Some(Snap::Struct { layers }) => {
                self.layers = layers;
                true
            }
        }
    }

    pub fn composite(&self) -> Layer {
        composite(&self.layers)
    }
}

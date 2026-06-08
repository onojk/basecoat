//! Verify that run_kaleido's multi-layer composite path genuinely combines
//! ALL marked layers into the kaleido source — not just one of them.

use basecoat::kaleido::kaleido;
use basecoat::layers::{composite, BlendMode, Layer};
use std::f64::consts::TAU;

const SIZE: usize = 64;

fn solid_layer(r: f32, g: f32, b: f32, a: f32) -> Layer {
    let rgba = (0..SIZE * SIZE * 4)
        .map(|i| match i % 4 { 0 => r, 1 => g, 2 => b, _ => a })
        .collect();
    Layer { rgba, width: SIZE as u32, height: SIZE as u32,
            mode: BlendMode::Normal, opacity: 1.0, visible: true, locked: false, name: String::new() }
}

fn spot_layer(px: usize, py: usize, r: f32, g: f32, b: f32) -> Layer {
    let mut layer = solid_layer(0.0, 0.0, 0.0, 0.0);
    let i = (py * SIZE + px) * 4;
    layer.rgba[i    ] = r;
    layer.rgba[i + 1] = g;
    layer.rgba[i + 2] = b;
    layer.rgba[i + 3] = 1.0;
    layer
}

fn get_pixel(layer: &Layer, x: usize, y: usize) -> [f32; 4] {
    let i = (y * SIZE + x) * 4;
    [layer.rgba[i], layer.rgba[i+1], layer.rgba[i+2], layer.rgba[i+3]]
}

fn test_composite_includes_both_layers() {
    // Layer 0 (bottom): solid red, fully opaque
    // Layer 1 (top): single white pixel at (50, 32) on transparent background
    //   Position (50, 32) → dx = 50 - 31.5 = 18.5, dy = 32 - 31.5 = 0.5
    //   angle ≈ atan2(0.5, 18.5) ≈ 1.5° — inside first wedge [0°, 60°) for segments=6
    let color_layer = solid_layer(1.0, 0.0, 0.0, 1.0);
    let spot_x = 50usize;
    let spot_y = 32usize;
    let lines_layer = spot_layer(spot_x, spot_y, 1.0, 1.0, 1.0); // white spot

    // Direct composite of [color, lines] — this is what run_kaleido does
    let composited = composite(&[color_layer, lines_layer]);

    // Verify composite at spot pixel is white (lines layer covered color)
    let spot_px = get_pixel(&composited, spot_x, spot_y);
    assert!(
        (spot_px[0] - 1.0).abs() < 0.01 && (spot_px[1] - 1.0).abs() < 0.01 && spot_px[3] > 0.99,
        "composite at spot should be white: got {spot_px:?}"
    );

    // Verify composite elsewhere is red (color layer shows through transparent)
    let elsewhere = get_pixel(&composited, 10, 10);
    assert!(
        (elsewhere[0] - 1.0).abs() < 0.01 && elsewhere[1] < 0.01 && elsewhere[3] > 0.99,
        "composite elsewhere should be red: got {elsewhere:?}"
    );

    println!("PASS  composite: spot={spot_px:?}, elsewhere={elsewhere:?}");

    // Now kaleido the composite — the white spot is in the first wedge, so the
    // kaleido output should have a near-white pixel at the same location
    let out = kaleido(&composited, 6, 0.0, 1.0);

    // The spot at (50, 32) is near angle 0° so it maps back to itself
    let cx = (SIZE as f64 - 1.0) / 2.0;
    let cy = cx;
    let dx = spot_x as f64 - cx;
    let dy = spot_y as f64 - cy;
    let r = dx.hypot(dy);
    let a = dy.atan2(dx);
    let wedge_w = TAU / 6.0;
    let k = (a / wedge_w).floor();
    let am = (a - k * wedge_w).clamp(0.0, wedge_w);
    let am_fold = if (k as i64).rem_euclid(2) == 1 { wedge_w - am } else { am };
    let src_x = cx + (r / 1.0) * am_fold.cos();
    let src_y = cy + (r / 1.0) * am_fold.sin();
    println!("  spot at ({spot_x},{spot_y}): angle={:.2}°, maps back to src ({src_x:.2},{src_y:.2})",
             a.to_degrees());

    // The source coordinate for the spot pixel should be very close to the spot
    let dist = ((src_x - spot_x as f64).powi(2) + (src_y - spot_y as f64).powi(2)).sqrt();
    assert!(dist < 1.5, "spot maps back to src ({src_x:.2},{src_y:.2}), expected near ({spot_x},{spot_y}), dist={dist:.3}");

    // The kaleido output at (spot_x, spot_y) should be near-white (sampled from composite's white spot)
    let out_spot = get_pixel(&out, spot_x, spot_y);
    assert!(
        out_spot[0] > 0.9 && out_spot[1] > 0.9 && out_spot[3] > 0.99,
        "kaleido output at spot should be near-white (both layers' content present): got {out_spot:?}"
    );

    // Most other pixels should be near-red (from the bottom color layer)
    let out_elsewhere = get_pixel(&out, 10, 10);
    assert!(
        out_elsewhere[0] > 0.9 && out_elsewhere[1] < 0.1,
        "kaleido output elsewhere should be red (color layer content): got {out_elsewhere:?}"
    );

    println!("PASS  kaleido: spot={out_spot:?}, elsewhere={out_elsewhere:?}");
    println!("PASS  test_composite_includes_both_layers");
}

fn test_composite_order_bottom_first() {
    // Verify that ascending index = bottom of composite
    // Layer 0 (bottom): green, opaque
    // Layer 1 (top): red, opaque — should WIN over green everywhere
    let green = solid_layer(0.0, 1.0, 0.0, 1.0);
    let red   = solid_layer(1.0, 0.0, 0.0, 1.0);
    let result = composite(&[green, red]);
    let px = get_pixel(&result, 32, 32);
    assert!(
        (px[0] - 1.0).abs() < 0.01 && px[1] < 0.01,
        "top (index 1) red layer should cover bottom green: got {px:?}"
    );
    println!("PASS  test_composite_order_bottom_first (ascending=bottom-first ✓)");
}

fn test_transparent_top_shows_bottom() {
    // Verify that transparent top layer lets bottom show through
    // Layer 0 (bottom): solid blue, opaque
    // Layer 1 (top): transparent everywhere — bottom should show through
    let blue  = solid_layer(0.0, 0.0, 1.0, 1.0);
    let trans = solid_layer(0.0, 0.0, 0.0, 0.0);
    let result = composite(&[blue, trans]);
    let px = get_pixel(&result, 32, 32);
    assert!(
        px[2] > 0.99 && (px[0] + px[1]) < 0.01,
        "transparent top should let blue bottom show through: got {px:?}"
    );
    println!("PASS  test_transparent_top_shows_bottom");
}

fn test_invisible_mark_force_visible() {
    // Regression: if one marked layer is invisible, composite() silently skips it.
    // run_kaleido now force-sets visible=true on all cloned marks so all N
    // contribute. This test verifies the force-visible logic.
    let color = solid_layer(1.0, 0.0, 0.0, 1.0); // red, visible
    let mut lines = spot_layer(50, 32, 0.0, 0.0, 0.0); // black spot
    lines.visible = false; // invisible — would be silently skipped without the fix

    // Without fix: composite skips invisible layer → only red
    let without_fix = composite(&[color.clone(), lines.clone()]);
    let spot_without = get_pixel(&without_fix, 50, 32);
    assert!(
        spot_without[0] > 0.9 && spot_without[1] < 0.1,
        "without fix, invisible layer is skipped → spot stays red: {spot_without:?}"
    );

    // With fix (force visible=true on all marks before compositing):
    let mut lines_forced = lines.clone();
    lines_forced.visible = true;
    let with_fix = composite(&[color.clone(), lines_forced]);
    let spot_with = get_pixel(&with_fix, 50, 32);
    assert!(
        spot_with[0] < 0.1 && spot_with[1] < 0.1 && spot_with[3] > 0.99,
        "with fix, force-visible mark contributes → spot is black: {spot_with:?}"
    );

    println!("PASS  test_invisible_mark_force_visible");
}

fn main() {
    test_composite_order_bottom_first();
    test_transparent_top_shows_bottom();
    test_composite_includes_both_layers();
    test_gradient_plus_edge_kaleido();
    test_invisible_mark_force_visible();
    println!("All composite tests passed.");
}

fn gradient_rgba(size: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; size * size * 4];
    for row in 0..size {
        for col in 0..size {
            let i = (row * size + col) * 4;
            buf[i    ] = col as f32 / (size as f32 - 1.0);
            buf[i + 1] = row as f32 / (size as f32 - 1.0);
            buf[i + 2] = 0.5;
            buf[i + 3] = 1.0;
        }
    }
    buf
}

fn test_gradient_plus_edge_kaleido() {
    use basecoat::edge::edge;
    let size = 64usize;
    let grad_rgba = gradient_rgba(size);

    // Layer 0 (bottom): gradient, fully opaque
    let color_layer = Layer {
        rgba:    grad_rgba.clone(),
        width:   size as u32, height: size as u32,
        mode:    BlendMode::Normal, opacity: 1.0, visible: true, locked: false, name: String::new(),
    };

    // Layer 1 (top): edge layer from gradient — opaque black lines, transparent elsewhere
    let edge_buf = edge(&grad_rgba, size, size, 50.0, 1);
    let lines_layer = Layer {
        rgba:    edge_buf.clone(),
        width:   size as u32, height: size as u32,
        mode:    BlendMode::Normal, opacity: 1.0, visible: true, locked: false, name: String::new(),
    };

    // Count edge pixels in lines_layer
    let n_edge_pixels = edge_buf.chunks(4).filter(|c| c[3] > 0.5).count();
    println!("  gradient edge pixels: {n_edge_pixels}");
    assert!(n_edge_pixels > 0, "edge layer should have some opaque black pixels");

    // Composite (bottom=color, top=lines) — this is what run_kaleido does
    let composited = composite(&[color_layer, lines_layer]);

    // Count black pixels in composite (edge pixels should show through)
    let black_in_comp = composited.rgba.chunks(4)
        .filter(|c| c[0] < 0.01 && c[1] < 0.01 && c[2] < 0.01 && c[3] > 0.99)
        .count();
    // Count non-black (plasma) pixels in composite  
    let color_in_comp = composited.rgba.chunks(4)
        .filter(|c| c[3] > 0.99 && (c[0] > 0.1 || c[1] > 0.1 || c[2] > 0.1))
        .count();
    println!("  composite: black={black_in_comp}, color={color_in_comp}");
    assert!(black_in_comp > 0, "composite should have black edge pixels from lines_layer");
    assert!(color_in_comp > 0, "composite should have color pixels from color_layer");

    // Apply kaleido to composite — verify output has BOTH black and color pixels
    let out = kaleido(&composited, 6, 0.0, 1.0);
    let black_in_out = out.rgba.chunks(4)
        .filter(|c| c[0] < 0.05 && c[1] < 0.05 && c[2] < 0.05 && c[3] > 0.99)
        .count();
    let color_in_out = out.rgba.chunks(4)
        .filter(|c| c[3] > 0.99 && (c[0] > 0.1 || c[1] > 0.1))
        .count();
    println!("  kaleido output: black={black_in_out}, color={color_in_out}");
    assert!(black_in_out > 0, "kaleido output should have black pixels (from lines_layer)");
    assert!(color_in_out > 0, "kaleido output should have color pixels (from color_layer)");

    println!("PASS  test_gradient_plus_edge_kaleido");
}

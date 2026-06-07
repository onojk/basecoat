//! Verify that the mark→kaleido index mapping is correct for every layer position.
//! Uses solid-color layers so kaleido output is unambiguously identifiable.

use basecoat::kaleido::kaleido;
use basecoat::layers::{composite, BlendMode, Layer};

const SZ: u32 = 32;

fn solid(r: f32, g: f32, b: f32) -> Layer {
    let n = (SZ * SZ * 4) as usize;
    let rgba = (0..n).map(|i| match i % 4 { 0 => r, 1 => g, 2 => b, _ => 1.0 }).collect();
    Layer { rgba, width: SZ, height: SZ, mode: BlendMode::Normal,
            opacity: 1.0, visible: true, name: String::new() }
}

fn dominant_channel(layer: &Layer) -> &'static str {
    let len = (SZ * SZ) as usize;
    let (mut sr, mut sg, mut sb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..len {
        let b = i * 4;
        sr += layer.rgba[b    ] as f64;
        sg += layer.rgba[b + 1] as f64;
        sb += layer.rgba[b + 2] as f64;
    }
    if sr >= sg && sr >= sb { "red" }
    else if sg >= sr && sg >= sb { "green" }
    else { "blue" }
}

// Simulate the GUI's mark→kaleido logic given a set of marked indices.
fn run_kaleido_for_marks(layers: &[Layer], marks: &[usize]) -> Vec<(usize, Layer)> {
    let mut sorted = marks.to_vec();
    sorted.sort_unstable();
    sorted.iter().map(|&i| {
        let mut src = layers[i].clone();
        src.visible = true;
        let out = kaleido(&src, 6, 0.0, 1.0);
        (i, out)
    }).collect()
}

fn test_single_mark_bottom() {
    // Stack: [red(0), green(1), blue(2)].  Mark bottom (stack idx 0).
    let layers = vec![solid(1.0, 0.0, 0.0), solid(0.0, 1.0, 0.0), solid(0.0, 0.0, 1.0)];
    let outputs = run_kaleido_for_marks(&layers, &[0]);
    assert_eq!(outputs.len(), 1);
    let (src_idx, ref out) = outputs[0];
    assert_eq!(src_idx, 0, "source index must be 0 (bottom)");
    assert_eq!(dominant_channel(out), "red",
        "kaleido of red layer must produce red output, got dominant={}", dominant_channel(out));
    println!("PASS  mark bottom (idx=0) → kaleido of red layer");
}

fn test_single_mark_middle() {
    // Stack: [red(0), green(1), blue(2)].  Mark middle (stack idx 1).
    let layers = vec![solid(1.0, 0.0, 0.0), solid(0.0, 1.0, 0.0), solid(0.0, 0.0, 1.0)];
    let outputs = run_kaleido_for_marks(&layers, &[1]);
    assert_eq!(outputs.len(), 1);
    let (src_idx, ref out) = outputs[0];
    assert_eq!(src_idx, 1, "source index must be 1 (middle)");
    assert_eq!(dominant_channel(out), "green",
        "kaleido of green layer must produce green output, got dominant={}", dominant_channel(out));
    println!("PASS  mark middle (idx=1) → kaleido of green layer");
}

fn test_single_mark_top() {
    // Stack: [red(0), green(1), blue(2)].  Mark top (stack idx 2).
    let layers = vec![solid(1.0, 0.0, 0.0), solid(0.0, 1.0, 0.0), solid(0.0, 0.0, 1.0)];
    let outputs = run_kaleido_for_marks(&layers, &[2]);
    assert_eq!(outputs.len(), 1);
    let (src_idx, ref out) = outputs[0];
    assert_eq!(src_idx, 2, "source index must be 2 (top)");
    assert_eq!(dominant_channel(out), "blue",
        "kaleido of blue layer must produce blue output, got dominant={}", dominant_channel(out));
    println!("PASS  mark top (idx=2) → kaleido of blue layer");
}

fn test_two_marks_bottom_and_top() {
    // Stack: [red(0), green(1), blue(2)].  Mark bottom AND top.
    let layers = vec![solid(1.0, 0.0, 0.0), solid(0.0, 1.0, 0.0), solid(0.0, 0.0, 1.0)];
    let outputs = run_kaleido_for_marks(&layers, &[0, 2]);
    assert_eq!(outputs.len(), 2, "two marks → two outputs");
    // sorted order: [0, 2] → outputs[0]=kaleido(red), outputs[1]=kaleido(blue)
    assert_eq!(outputs[0].0, 0);
    assert_eq!(dominant_channel(&outputs[0].1), "red",
        "first output must be kaleido of red (idx=0)");
    assert_eq!(outputs[1].0, 2);
    assert_eq!(dominant_channel(&outputs[1].1), "blue",
        "second output must be kaleido of blue (idx=2)");
    println!("PASS  marks {{0,2}} → two outputs: red then blue");
}

fn test_four_layer_mark_second_from_top() {
    // 4-layer stack, mark the second from top (stack idx 2).
    // In the panel this would appear as the SECOND ROW from the top.
    let layers = vec![
        solid(1.0, 0.0, 0.0),  // idx 0: red   (bottom, panel row 3)
        solid(0.0, 1.0, 0.0),  // idx 1: green (panel row 2)
        solid(0.0, 0.5, 1.0),  // idx 2: cyan  (panel row 1) ← marked
        solid(0.0, 0.0, 1.0),  // idx 3: blue  (top, panel row 0)
    ];
    let outputs = run_kaleido_for_marks(&layers, &[2]);
    assert_eq!(outputs.len(), 1);
    let (src_idx, ref out) = outputs[0];
    assert_eq!(src_idx, 2, "source must be idx=2 (second from top in panel)");
    // cyan-ish: blue and green both > 0.4, red near 0
    let n = (SZ * SZ) as usize;
    let (mut sr, mut sg, mut sb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let b = i * 4;
        sr += out.rgba[b    ] as f64;
        sg += out.rgba[b + 1] as f64;
        sb += out.rgba[b + 2] as f64;
    }
    let pix = n as f64;
    let (mr, mg, mb) = (sr / pix, sg / pix, sb / pix);
    assert!(mr < 0.1, "red channel should be near 0 (cyan source): got {mr:.3}");
    assert!(mg > 0.4, "green channel should be ~0.5 (cyan source): got {mg:.3}");
    assert!(mb > 0.8, "blue channel should be ~1.0 (cyan source): got {mb:.3}");
    println!("PASS  4-layer stack: mark idx=2 (panel row 1) → kaleido of cyan layer (r={mr:.2} g={mg:.2} b={mb:.2})");
}

fn test_merge_marks_non_top() {
    // Merge marks {0, 1} in a 3-layer stack.
    // Result should be composite(red, green) ≈ green (top mark covers bottom).
    let red   = solid(1.0, 0.0, 0.0);
    let green = solid(0.0, 1.0, 0.0);
    let blue  = solid(0.0, 0.0, 1.0);

    let marks = [0usize, 1];
    let layers_to_merge: Vec<Layer> = marks.iter()
        .map(|&i| [red.clone(), green.clone(), blue.clone()][i].clone())
        .collect();
    let merged = composite(&layers_to_merge);

    // green (idx 1) on top of red (idx 0) → green wins
    assert_eq!(dominant_channel(&merged), "green",
        "merge of [red(0), green(1)] should be green: got {}", dominant_channel(&merged));
    println!("PASS  merge marks {{0,1}} → composite(red,green) = green");
}

fn test_merge_skips_unmarked_top() {
    // Mark bottom and middle but NOT top.  Merged result should have no blue.
    let layers = [solid(1.0, 0.0, 0.0), solid(0.0, 1.0, 0.0), solid(0.0, 0.0, 1.0)];
    let marks = [0usize, 1]; // not 2
    let layers_to_merge: Vec<Layer> = marks.iter()
        .map(|&i| layers[i].clone())
        .collect();
    let merged = composite(&layers_to_merge);

    let n = (SZ * SZ) as usize;
    let sb: f64 = (0..n).map(|i| merged.rgba[i * 4 + 2] as f64).sum();
    assert!(sb < 1.0, "merged output must have no blue (top layer idx=2 not in marks): total_blue={sb:.3}");
    println!("PASS  merge marks {{0,1}} (not 2) → no blue in merged output");
}

fn main() {
    test_single_mark_bottom();
    test_single_mark_middle();
    test_single_mark_top();
    test_two_marks_bottom_and_top();
    test_four_layer_mark_second_from_top();
    test_merge_marks_non_top();
    test_merge_skips_unmarked_top();
    println!("All mark-kaleido mapping tests passed.");
}

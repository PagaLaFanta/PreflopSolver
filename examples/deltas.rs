use preflop_engine::eval::board_bunch::BoardBunching;
use preflop_engine::io::ranges::parse_range;
fn main() {
    let r: Vec<Vec<f32>> = std::env::args().skip(1).map(|s| { let mut v = parse_range(&s).unwrap(); v.resize(1328, 0.0); v }).collect();
    let d = BoardBunching::deck_deltas(r.iter().map(|v| &v[..])).unwrap();
    println!("{:?}", d.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
}

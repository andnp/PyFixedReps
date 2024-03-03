use numpy::{Ix1, ndarray::{Array1, s, ArrayView2, ArrayView1}};

use crate::BoundStrat;

fn get_axis_cell(x: f64, tiles: u32, bound_strat: &BoundStrat) -> u32 {
    let t = f64::from(tiles);
    // NOTE: this assumes x is positive by the time we get here
    let i = f64::floor(x * t) as u32;

    match bound_strat {
        BoundStrat::Clip => i.clamp(0, tiles - 1),
        BoundStrat::Wrap => i % (tiles - 1),
    }

}

fn get_tiling_index(dims: u32, tiles_per_tiling: u32, tiles_per_dim: &ArrayView1<u32>, bound_strats: &Vec<BoundStrat>, pos: &Array1<f64>) -> u32 {
    let mut ind = 0;
    let mut already_seen: u32 = 1;

    for d in 0..dims {
        let idx = Ix1(d as usize);
        let x = *pos.get(idx).expect("Index out-of-bounds for numpy array");
        let tiles = *tiles_per_dim.get(idx).expect("Index out-of-bounds");
        let strat = &bound_strats[d as usize];

        let axis = get_axis_cell(x, tiles, strat);
        ind += axis * already_seen;
        already_seen *= tiles;
    }

    ind % tiles_per_tiling
}

pub fn get_tc_indices(dims: u32, tiles: &ArrayView1<u32>, tilings: u32, offsets: &ArrayView2<f64>, bound_strats: &Vec<BoundStrat>, pos: &Array1<f64>) -> Array1<u32> {
    let tiles_per_tiling = tiles.product();
    let mut index = Array1::zeros(tilings as usize);

    for ntl in 0..tilings {
        let off = offsets.slice(s![ntl as usize, ..]);
        let arr = pos + &off;
        let ind = get_tiling_index(dims, tiles_per_tiling, tiles, bound_strats, &arr);
        index[ntl as usize] = ind + tiles_per_tiling * ntl;
    }

    index
}

// ----------------
// -- Unit tests --
// ----------------
#[cfg(test)]
mod tests {
    use numpy::ndarray::Array1;

    use crate::BoundStrat;

    #[test]
    fn get_axis_cell() {
        // check endpoints
        let res = super::get_axis_cell(0.0, 8, &BoundStrat::Clip);
        assert_eq!(res, 0);

        let res = super::get_axis_cell(1.0, 8, &BoundStrat::Clip);
        assert_eq!(res, 7);

        // check out-of-bounds
        let res = super::get_axis_cell(-0.01, 8, &BoundStrat::Clip);
        assert_eq!(res, 0);

        let res = super::get_axis_cell(1.03, 8, &BoundStrat::Clip);
        assert_eq!(res, 7);

        // check middle
        let res = super::get_axis_cell(0.124, 8, &BoundStrat::Clip);
        assert_eq!(res, 0);
        let res = super::get_axis_cell(0.126, 8, &BoundStrat::Clip);
        assert_eq!(res, 1);
        let res = super::get_axis_cell(0.249, 8, &BoundStrat::Clip);
        assert_eq!(res, 1);
        let res = super::get_axis_cell(0.26, 8, &BoundStrat::Clip);
        assert_eq!(res, 2);
    }

    #[test]
    fn get_tiling_index() {
        let arr = Array1::from_iter([0.1]);
        let strats = vec![BoundStrat::Clip];
        let tiles: Array1<u32> = Array1::from_iter([8]);
        let res = super::get_tiling_index(1, 8, &tiles.view(), &strats, &arr);
        assert_eq!(res, 0);

        let arr = Array1::from_iter([0.1, 0.1]);
        let strats = vec![BoundStrat::Clip, BoundStrat::Clip];
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, &tiles.view(), &strats, &arr);
        assert_eq!(res, 0);

        let arr = Array1::from_iter([0.126, 0.1]);
        let strats = vec![BoundStrat::Clip, BoundStrat::Clip];
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, &tiles.view(), &strats, &arr);
        assert_eq!(res, 1);

        let arr = Array1::from_iter([0.126, 0.126]);
        let strats = vec![BoundStrat::Clip, BoundStrat::Clip];
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, &tiles.view(), &strats, &arr);
        assert_eq!(res, 9);

        let arr = Array1::from_iter([1.0, 1.0]);
        let strats = vec![BoundStrat::Clip, BoundStrat::Clip];
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, &tiles.view(), &strats, &arr);
        assert_eq!(res, 63);

        let arr = Array1::from_iter([0.21, 1.0]);
        let strats = vec![BoundStrat::Clip, BoundStrat::Clip];
        let tiles: Array1<u32> = Array1::from_iter([5, 8]);
        let res = super::get_tiling_index(2, 40, &tiles.view(), &strats, &arr);
        assert_eq!(res, 36);
    }
}

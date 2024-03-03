use std::str::FromStr;

use numpy::ndarray::{ArrayView1, ArrayView2, Array1, Zip};
use numpy::{IntoPyArray, PyReadonlyArray2};
use numpy::{PyArray1,PyReadonlyArray1};
use pyo3::prelude::*;

mod tc;

pub enum BoundStrat {
    Clip,
    Wrap,
}

impl FromStr for BoundStrat {
    type Err = ();

    fn from_str(input: &str) -> Result<BoundStrat, Self::Err> {
        match input {
            "clip" => Ok(BoundStrat::Clip),
            "wrap" => Ok(BoundStrat::Wrap),
            _ => Err(()),
        }
    }
}

fn minmax_scale(pos: f64, bound: ArrayView1<f64>) -> f64 {
    (pos - bound[0]) / (bound[1] - bound[0])
}

fn apply_bounds(pos: ArrayView1<f64>, bounds: ArrayView2<f64>) -> Array1<f64> {
    let mut out: Array1<f64> = Array1::zeros(pos.raw_dim());

    Zip::from(&mut out)
        .and(&pos)
        .and(bounds.rows())
        .for_each(|o, &p, b| {
            *o = minmax_scale(p, b);
        });

    out
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name="get_tc_indices")]
    fn test_py<'py>(
        py: Python<'py>,
        dims: u32,
        tiles: PyReadonlyArray1<u32>,
        tilings: u32,
        bounds: PyReadonlyArray2<f64>,
        offsets: PyReadonlyArray2<f64>,
        bound_strats: Vec<&str>,
        pos: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<u32> {
        let offsets = offsets.as_array();
        let pos = pos.as_array();
        let bounds = bounds.as_array();
        let tiles = tiles.as_array();
        let bound_strats: Vec<BoundStrat> =
            bound_strats
                .iter()
                .map(|v| BoundStrat::from_str(&v).expect("Unknown bounding strategy!"))
                .collect();

        let pos = apply_bounds(pos, bounds);
        let res = py.allow_threads(|| tc::get_tc_indices(dims, &tiles, tilings, &offsets, &bound_strats, &pos));
        res.into_pyarray(py)
    }

    Ok(())
}

use rand::Rng;

#[derive(Clone)]
pub struct NetworkLayer {
    pub num_nodes: usize,
    pub activations: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub changes: Vec<Vec<f64>>,
}

impl NetworkLayer {
    pub fn new(num_nodes: usize, next: Option<&NetworkLayer>) -> NetworkLayer {
        let mut res = NetworkLayer {
            num_nodes,
            activations: vec![0_f64; num_nodes],
            weights: vec![Vec::new(); num_nodes],
            changes: vec![Vec::new(); num_nodes],
        };

        if next.is_none() {
            return res;
        }

        let next_num_nodes = next.unwrap().num_nodes;

        res.weights.iter_mut()
            .for_each(|v| v.resize(next_num_nodes, rand::thread_rng().gen_range(-1.0..1.0)));

        res.changes.iter_mut()
            .for_each(|v| v.resize(next_num_nodes, 0_f64));

        res
    }
}

mod nn;

fn main() {
    const NUM_INPUTS: usize = 2;
    const NUM_HIDDENS: usize = 2;
    const NUM_OUTPUTS: usize = 1;

    let mut network = nn::nn::NeuralNetwork::new(NUM_INPUTS, NUM_HIDDENS, NUM_OUTPUTS);

    let inputs = vec![
        vec![0_f64, 0_f64], vec![0_f64, 1_f64],
        vec![1_f64, 0_f64], vec![1_f64, 1_f64],
    ];

    let outputs = vec![
        vec![0_f64], vec![1_f64],
        vec![1_f64], vec![0_f64],
    ];

    let trained = network.train(inputs.clone(), outputs.clone(), 100000, 0.3, 0.6);
    println!("E_1 = {} E_2 = {} %learned = {}", trained[0], trained[trained.len() - 1], 100_f64 * trained[0] / trained[trained.len() - 1]);

    println!("XOR(1, 1) = {} (expected 0)", network.activate(vec![1_f64, 1_f64])[0]);
}

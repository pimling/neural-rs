use crate::nn::util;

pub struct NeuralNetwork {
    input: util::NetworkLayer,
    hidden: util::NetworkLayer,
    output: util::NetworkLayer,
}

impl NeuralNetwork {
    pub fn new(num_inputs: usize, num_hiddens: usize, num_outputs: usize) -> NeuralNetwork {
        let output = util::NetworkLayer::new(num_outputs, Option::None);
        let hidden = util::NetworkLayer::new(num_hiddens + 1, Option::Some(&output));
        let input = util::NetworkLayer::new(num_inputs + 1, Option::Some(&hidden));

        NeuralNetwork {
            input,
            hidden,
            output,
        }
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1_f64 / (1_f64 + std::f64::consts::E.powf(-x))
    }

    fn sigmoid_gradient(&self, y: f64) -> f64 {
        return y * (1_f64 - y);
    }

    // deltaW_ij = -n * dE/dW_ij
    fn run_backpropagation(&mut self, target: Vec<f64>, learn: f64, momentum: f64) -> f64 {
        if target.len() != self.output.num_nodes as usize {
            return -1.0;
        }

        // calculate the difference in output layer to given target
        let output_offset = (0_usize..self.output.num_nodes)
            .map(|i| self.sigmoid_gradient(self.output.activations[i]) * (target[i] - self.output.activations[i]))
            .collect::<Vec<f64>>();

        // repeat for hidden layer
        let hidden_offset = (0_usize..self.hidden.num_nodes)
            .map(|i| self.sigmoid_gradient(self.hidden.activations[i]) * (0..self.output.num_nodes)
                .map(|j| self.hidden.weights[i][j] * output_offset[j])
                .sum::<f64>())
            .collect::<Vec<f64>>();

        // update hidden weights and changes accordingly
        for i in 0_usize..self.hidden.num_nodes {
            for j in 0_usize..self.output.num_nodes {
                self.hidden.weights[i][j] = self.hidden.weights[i][j] + learn * output_offset[j] * self.hidden.activations[i] + momentum * self.hidden.changes[i][j];
                self.hidden.changes[i][j] = output_offset[j] * self.hidden.activations[i];
            }
        }

        // same for the input layer
        for i in 0_usize..self.input.num_nodes {
            for j in 0_usize..self.hidden.num_nodes {
                self.input.weights[i][j] = self.input.weights[i][j] + learn * hidden_offset[j] * self.input.activations[i] + momentum * self.input.changes[i][j];
                self.input.changes[i][j] = hidden_offset[j] * self.input.activations[i];
            }
        }

        // sum of euclidean distances
        (0..target.len())
            .map(|i| 0.5_f64 * (target[i] - self.output.activations[i]).powi(2))
            .sum::<f64>()
    }

    pub fn activate(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.input.num_nodes - 1 as usize {
            return Vec::new();
        }

        // set input layer activations to given inputs
        for i in 0..self.input.num_nodes - 1 {
            self.input.activations[i] = inputs[i];
        }

        // calculate hidden layer(s) activations using inputs and input weights
        self.hidden.activations = (0_usize..self.hidden.num_nodes)
            .map(|i| (0_usize..self.input.num_nodes)
                .map(|j| self.input.activations[j] * self.input.weights[j][i])
                .sum::<f64>())
            .map(|x| self.sigmoid(x))
            .collect::<Vec<f64>>();

        // repeat for output layer
        self.output.activations = (0_usize..self.output.num_nodes)
            .map(|i| (0_usize..self.hidden.num_nodes)
                .map(|j| self.hidden.activations[j] * self.hidden.weights[j][i])
                .sum::<f64>())
            .map(|x| self.sigmoid(x))
            .collect::<Vec<f64>>();

        self.output.activations.clone()
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>, num_iter: usize, learn: f64, momentum: f64) -> Vec<f64> {
        if inputs.len() != outputs.len() {
            return Vec::new();
        }

        // for each iteration activate the network, run the backpropagation formula and sum the errors
        // for each input. then return the vector of size equal to num_iter
        (0..num_iter)
            .map(|_i| (0..inputs.len())
                .map(|j| {
                    self.activate(inputs[j].clone());
                    self.run_backpropagation(outputs[j].clone(), learn, momentum)
                })
                .sum::<f64>())
            .collect::<Vec<f64>>()
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private int _inputsCount;
        private NeuronLayer _lastLayer;

        private List<NeuronLayer> _layers = new List<NeuronLayer>();

        public NeuralNetwork(int inputsCount, int outputCount, int layersCount, int layerSize)
        {
            _lastLayer = new NeuronLayer(outputCount);
            _lastLayer.AddInputs(layerSize);

            _inputsCount = inputsCount;

            var firstLayer = new NeuronLayer(layerSize);
            firstLayer.AddInputs(inputsCount);
            _layers.Add(firstLayer);

            for (var i = 1; i < layersCount; i++)
            {
                var nl = new NeuronLayer(layerSize);
                nl.AddInputs(layerSize);
                _layers.Add(nl);
            }
        }

        public double LearningRate { get; set; }

        public static double ActivationFunction(double x) =>
            1.0 / (1.0 + Math.Exp(-x));

        public static double ActivationFunctionDerivative(double x) =>
            ActivationFunction(x) * (1 - ActivationFunction(x));

        public List<double> Calculate(List<double> input)
        {
            if (input.Count != _inputsCount)
                throw new Exception("Inputs count does not match NN inputs.");
            var currInput = input;
            foreach (var layer in _layers)
            {
                currInput = layer.Calculate(currInput);
            }

            return _lastLayer.Calculate(currInput, true);
        }


        public override string ToString()
        {
            var sb = new StringBuilder();

            for (int i = 0; i < _layers.Count; i++)
            {
                sb.Append($"Layer {i}:\n").Append(_layers[i]);
            }

            sb.Append("Output layer:\n").Append(_lastLayer);
            return sb.ToString();
        }

        public void Train(Tuple<List<double>, List<double>> datapoint)
        {
            // Predict result
            var predictedOutput = Calculate(datapoint.Item1);

            // Calculate errors in last layer
            for (int i = 0; i < _lastLayer.Neurons.Count; i++)
            {
                _lastLayer.Neurons[i].Error = (datapoint.Item2[i] - predictedOutput[i]);
                _lastLayer.Neurons[i].Delta = _lastLayer.Neurons[i].Error;
            }

            // Calculate errors in hidden layers
            for (var currentLayerIndex = _layers.Count - 1; currentLayerIndex >= 0; currentLayerIndex--)
            {
                var currLayer = _layers[currentLayerIndex];
                var nextLayer = (currentLayerIndex == _layers.Count - 1) ? _lastLayer : _layers[currentLayerIndex + 1];

                for (var i = 0; i < currLayer.Neurons.Count; i++)
                {
                    var currentError = nextLayer.Neurons.Sum(t => t.Delta * t.InputWeights[i]);

                    currLayer.Neurons[i].Error = currentError;
                    currLayer.Neurons[i].Delta =
                        currentError * ActivationFunctionDerivative(currLayer.Neurons[i].SumCache);
                }
            }

            // Change weights in last layer
            foreach (var n in _lastLayer.Neurons)
            {
                for (var j = 0; j < n.InputWeights.Count - 1; j++)
                {
                    n.InputWeights[j] += 2 * n.Delta * LearningRate * _lastLayer.InputCache[j];
                }

                n.InputWeights[n.InputWeights.Count - 1] = 2 * n.Delta * LearningRate;
            }

            // Change weights in hidden layers
            for (var currentLayerIndex = _layers.Count - 1; currentLayerIndex >= 0; currentLayerIndex--)
            {
                var currLayer = _layers[currentLayerIndex];

                foreach (var n in currLayer.Neurons)
                {
                    for (var j = 0; j < n.InputWeights.Count - 1; j++)
                    {
                        n.InputWeights[j] += 2 * n.Delta * LearningRate * currLayer.InputCache[j];
                    }

                    n.InputWeights[n.InputWeights.Count - 1] = 2 * n.Delta * LearningRate;
                }
            }
        }
    }
}
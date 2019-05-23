using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuronLayer
    {
        public List<Neuron> Neurons;

        public NeuronLayer(int size)
        {
            Neurons = new List<Neuron>(size);
            for (int i = 0; i < size; i++)
            {
                Neurons.Add(new Neuron());
            }
        }

        public List<double> Calculate(List<double> input, bool last = false)
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(!last ? neuron.Calculate(input) : neuron.CalculateLast(input));
            }

            return result;
        }

        public void AddInputs(int count)
        {
            foreach (var n in Neurons)
            {
                n.AddInputs(count);
            }
        }

        public override string ToString()
        {
            var sb = new StringBuilder();

            for (int i = 0; i < Neurons.Count; i++)
            {
                sb.Append($"\tNeuron {i}: ").Append(Neurons[i]).Append('\n');
            }

            return sb.ToString();
        }
    }
}
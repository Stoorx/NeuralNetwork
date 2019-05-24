using System;

namespace NeuralNetwork
{
    internal class Program
    {
        private static double TargetFunction(double x) => Math.Sqrt(1 + 4 * x + 12 * x * x);

        public static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(1, 10, 3, 10);
            Console.WriteLine(nn);

//            Dataset ds = new Dataset(50,
//                x => { return new List<double> {TargetFunction(x[0])}; },
//                new List<Tuple<double, double>> {new Tuple<double, double>(-10.0, 10.0)}
//            );
//            //Console.WriteLine(ds);
//
//            Console.WriteLine(new Dataset(ds.GetBatch(0.1)));
        }
    }
}
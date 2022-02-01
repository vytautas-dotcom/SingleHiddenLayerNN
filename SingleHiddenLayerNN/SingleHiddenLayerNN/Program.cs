using System;

namespace SingleHiddenLayerNN
{
    class Program
    {
        static void Main(string[] args)
        {
            Data data = new Data();
            data.Info();

            ShowData showData = new ShowData();

            Console.WriteLine("\n\t\tAll raw data\n");
            showData.Show(data.rawData, 5, 2, true);

            showData.SplitData(data.rawData, 37, out double[][] train, out double[][] test);

            Console.WriteLine("Train data:");
            showData.Show(train, 5, 2, true);
            Console.WriteLine("\nTest data:");
            showData.Show(test, 2, 2, true);

            Console.WriteLine("\n\n\nNeural Network of 4-input, 6-hidden, 3-output layers\n");

            NeuralNetwork network = new NeuralNetwork(4, 6, 3);

            Console.WriteLine("Using 12 particles and 500 iterations for estimashion of weights and biases\n");

            double[] bestWeights = network.Train(train, 12, 500);

            Console.WriteLine("Best weights and biases:\n");

            showData.ShowVector(bestWeights, 5, 3, true);

            Console.WriteLine("Testing best weights on test data:\n");

            network.SetWeights(bestWeights);

            double trainAcc = network.Accuracy(train);
            Console.WriteLine($"Accuracy on train data: {trainAcc}");

            double testAcc = network.Accuracy(test);
            Console.WriteLine($"Accuracy on test data: {testAcc}");


        }
    }
}

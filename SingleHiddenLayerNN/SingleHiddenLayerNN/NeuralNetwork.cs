using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleHiddenLayerNN
{
    class NeuralNetwork
    {
        private int numInput;         //number of nodes for input layer
        private int numHidden;        //number of nodes for hidden layer
        private int numOutput;        //number of nodes for output layer

        private double[] inputs;      //input values
        private double[][] ihWeights; //weights from input to hidden
        private double[] hBiases;     //biases of hidden
        private double[] hOutputs;    //outputs of hidden

        private double[][] hoWeights; //weights from hidden to output
        private double[] oBiases;     //biases of output
        private double[] outputs;     //outputs of output

        private Random rand;

        public NeuralNetwork(int numInput, int numHidden, int numOutput, int seed)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];

            ihWeights = MakeMatrix(numInput, numHidden);
            hBiases = new double[numHidden];
            hOutputs = new double[numHidden];

            hoWeights = MakeMatrix(numHidden, numOutput);
            oBiases = new double[numOutput];
            outputs = new double[numOutput];

            rand = new Random(seed);
        }

        private double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < result.Length; i++)
                result[i] = new double[cols];
            return result;
        }
    }
}

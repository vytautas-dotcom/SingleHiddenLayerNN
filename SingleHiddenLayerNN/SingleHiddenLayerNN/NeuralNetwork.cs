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

        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput + 1) * numHidden +
                             (numHidden + 1) * numOutput;

            if (weights.Length != numWeights)
                throw new Exception("Incorrect weights array length");

            int k = 0;

            //picks all weights and biases accordingly for hidden and output layers

            for (int i = 0; i < numInput; i++)
                for (int j = 0; j < numHidden; j++)
                    ihWeights[i][j] = weights[k++];

            for (int i = 0; i < numHidden; i++)
                hBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; i++)
                for (int j = 0; j < numOutput; j++)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; i++)
                oBiases[i] = weights[k++];
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];

            for (int i = 0; i < xValues.Length; i++)
                inputs[i] = xValues[i];

            for (int i = 0; i < numHidden; i++)
                for (int j = 0; j < numInput; j++)
                    hSums[i] += ihWeights[j][i] * inputs[j];

            for (int i = 0; i < numHidden; i++)
                hSums[i] += hBiases[i];

            for (int i = 0; i < numHidden; i++)
                hOutputs[i] = HyperTan(hSums[i]);

            for (int i = 0; i < numHidden; i++)
                for (int j = 0; j < numOutput; j++)
                    oSums[j] += hoWeights[j][i] * hOutputs[j];

            for (int i = 0; i < numOutput; i++)
                oSums[i] += oBiases[i];

            double[] softmaxOutputs = Softmax(oSums);
            Array.Copy(softmaxOutputs, outputs, softmaxOutputs.Length);

            double[] result = new double[numOutput];
            Array.Copy(outputs, result, result.Length);

            return result;
        }

        private double HyperTan(double v)
        {
            if (v < -20.0)
                return -1;
            else if (v > 20.0)
                return 1;
            else
                return Math.Tanh(v);
        }

        private double[] Softmax(double[] oSums)
        {
            double max = oSums[0];

            for (int i = 0; i < oSums.Length; i++)
                if(max < oSums[i]) max = oSums[i];

            double scale = 0.0;

            for (int i = 0; i < oSums.Length; i++)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];

            for (int i = 0; i < oSums.Length; i++)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result;
        }

        public double[] Train(double[][] trainData, int nParticles, int nEpochs)
        {
            int numWeights = (numInput + 1) * numHidden + (numHidden + 1) * numOutput;

            int epoch = 0;

            double minX = -10.0;
            double maxX = 10.0;
            double w = 0.729;
            double c1 = 1.49445;
            double c2 = 1.49445;
            double r1, r2;

            Particle[] swarm = new Particle[nParticles];
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleHiddenLayerNN
{
    class BackPropagationTrain
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

        //-------------------------
        private double[] hGrads;
        private double[] oGrads;

        private double[][] ihPrevWeightsDelta;
        private double[] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;

        private static Random rand;

        public BackPropagationTrain(int numInput, int numHidden, int numOutput)
        {
            rand = new Random(0);
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

            //-------------------------
            hGrads = new double[numHidden];
            oGrads = new double[numOutput];

            ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            hPrevBiasesDelta = new double[numHidden];
            hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            oPrevBiasesDelta = new double[numOutput];

            this.InitializeWeights();
        }

        static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < result.Length; i++)
                result[i] = new double[cols];
            return result;
        }

        private void InitializeWeights()
        {
            int numWeights = (1 + numInput) * numHidden + (1 + numHidden) * numOutput;

            double[] initialWeights = new double[numWeights];

            for (int i = 0; i < initialWeights.Length; i++)
                initialWeights[i] = 0.02 * rand.NextDouble() - 0.01;
            this.SetWeights(initialWeights);
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

            for (int i = 0; i < numOutput; i++)
                for (int j = 0; j < numHidden; j++)
                    oSums[i] += hoWeights[j][i] * hOutputs[j];

            for (int i = 0; i < numOutput; i++)
                oSums[i] += oBiases[i];

            double[] softmaxOutputs = Softmax(oSums);
            Array.Copy(softmaxOutputs, outputs, softmaxOutputs.Length);

            double[] result = new double[numOutput];
            Array.Copy(outputs, result, result.Length);

            return result;
        }


        static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else
                return Math.Tanh(v);
        }

        static double[] Softmax(double[] oSums)
        {
            double max = oSums[0];

            for (int i = 0; i < oSums.Length; i++)
                if (oSums[i] > max) max = oSums[i];

            double scale = 0.0;

            for (int i = 0; i < oSums.Length; i++)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];

            for (int i = 0; i < oSums.Length; i++)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result;
        }

        public void UpdateWeights(double[] targets, double alpha, double momentum)
        {
            //output gradients
            for (int i = 0; i < numOutput; i++)
                oGrads[i] = outputs[i] * (1 - outputs[i]) * (targets[i] - outputs[i]);
            //hidden gradients
            for (int i = 0; i < numHidden; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < numOutput; j++)
                    sum += (oGrads[j] * hoWeights[i][j]);
                hGrads[i] = sum * (1 - hOutputs[i]) * (1 + hOutputs[i]);
            }
            //hidden weights
            for (int i = 0; i < numInput; i++)
                for (int j = 0; j < numHidden; j++)
                {
                    double D = alpha * hGrads[j] * inputs[i];
                    ihWeights[i][j] += D;
                    ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                    ihPrevWeightsDelta[i][j] = D;
                }
            //hidden biases
            for (int i = 0; i < numHidden; i++)
            {
                double D = alpha * hGrads[i];
                hBiases[i] += D;
                hBiases[i] += momentum * hPrevBiasesDelta[i];
                hPrevBiasesDelta[i] = D;
            }
            //hidden output weights
            for (int i = 0; i < numHidden; i++)
                for (int j = 0; j < numOutput; j++)
                {
                    double D = alpha * oGrads[j] * hOutputs[i];
                    hoWeights[i][j] += D;
                    hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j];
                    hoPrevWeightsDelta[i][j] = D;
                }
            //output biases
            for (int i = 0; i < numOutput; i++)
            {
                double D = alpha * oGrads[i];
                oBiases[i] += D;
                oBiases[i] += momentum * oPrevBiasesDelta[i];
                oPrevBiasesDelta[i] = D;
            }

        }
        public void Train(double[][] train, int numEpoch, double alpha, double momentum)
        {
            int epoch = 0;
            double[] X = new double[numInput];  //inputs
            double[] T = new double[numOutput]; //targets

            int[] sequence = new int[train.Length];
            for (int i = 0; i < sequence.Length; i++)
                sequence[i] = i;

            while (epoch < numEpoch)
            {
                double mse = MeanSquareError(train);
                if (mse < 0.04)
                {
                    Console.WriteLine("epoch " + epoch);
                    Console.WriteLine("mse " + mse);
                    break; 
                }

                Shuffle(sequence);

                for (int i = 0; i < train.Length; i++)
                {
                    int index = sequence[i];
                    Array.Copy(train[index], X, numInput);
                    Array.Copy(train[index], numInput, T, 0, numOutput);
                    ComputeOutputs(X);

                    UpdateWeights(T, alpha, momentum);
                }
                epoch++;
            }
        }

        static void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; i++)
            {
                int index = rand.Next(i, sequence.Length);
                int temp = sequence[index];
                sequence[index] = sequence[i];
                sequence[i] = temp;
            }
        }
        private double MeanSquareError(double[][] trainData)
        {
            double[] X = new double[numInput];
            double[] T = new double[numOutput];

            double sumSquaredError = 0.0;

            for (int i = 0; i < trainData.Length; i++)
            {
                Array.Copy(trainData[i], X, numInput);              
                Array.Copy(trainData[i], numInput, T, 0, numOutput); 

                double[] Y = this.ComputeOutputs(X);
                for (int j = 0; j < Y.Length; j++)
                    sumSquaredError += ((T[j] - Y[j]) * (T[j] - Y[j]));
            }
            return sumSquaredError / trainData.Length;
        }

        public double Accuracy(double[][] testData)
        {
            int correct = 0;
            int wrong = 0;

            double[] xVal = new double[numInput];
            double[] tVal = new double[numInput];
            double[] yVal;

            for (int i = 0; i < testData.Length; i++)
            {
                Array.Copy(testData[i], xVal, numInput);
                Array.Copy(testData[i], numInput, tVal, 0, numOutput);

                yVal = this.ComputeOutputs(xVal);
                int indexOfMax = MaxIndex(yVal);

                if (tVal[indexOfMax] == 1.0)
                    correct++;
                else
                    wrong++;
            }
            return (correct * 1.0) / (correct + wrong);
        }

        static int MaxIndex(double[] yVal)
        {
            int indexOfMax = 0;
            double biggestVal = yVal[0];

            for (int i = 0; i < yVal.Length; i++)
            {
                if (yVal[i] > biggestVal)
                {
                    biggestVal = yVal[i];
                    indexOfMax = i;
                }
            }
            return indexOfMax;
        }
    }
}

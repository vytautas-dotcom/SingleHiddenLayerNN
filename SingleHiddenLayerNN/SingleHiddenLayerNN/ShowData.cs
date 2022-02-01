using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleHiddenLayerNN
{
    class ShowData
    {
        public void Show(double[][] rawData, int numOfFirstRows, int numOfLastRows, bool indices)
        {
            for (int i = 0; i < numOfFirstRows; i++)
            {
                if (indices)
                    Console.Write("[" + i.ToString().PadLeft(2) + "] ");
                for (int j = 0; j < rawData[i].Length; j++)
                    Console.Write(rawData[i][j].ToString().PadLeft(4) + " ");
                Console.WriteLine();
            }
            if (numOfFirstRows + numOfLastRows < rawData.Length)
                Console.WriteLine(". . .");
            for (int i = rawData.Length - numOfLastRows; i < rawData.Length; i++)
            {
                if (indices)
                    Console.Write("[" + i.ToString().PadLeft(2) + "] ");
                for (int j = 0; j < rawData[i].Length; j++)
                    Console.Write(rawData[i][j].ToString().PadLeft(4) + " ");
                Console.WriteLine();
            }
            Console.WriteLine("\n");
        }
        public void SplitData(double[][] rawData, int seed, out double[][] trainData, out double[][] testData)
        {
            Random rand = new Random(seed);
            int totalRows = rawData.Length;
            int trainRows = (int)(totalRows * 0.80);
            int testRows = totalRows - trainRows;

            trainData = new double[trainRows][];
            testData = new double[testRows][];

            double[][] copy = new double[rawData.Length][];

            for (int i = 0; i < copy.Length; i++)
                copy[i] = rawData[i];

            for (int i = 0; i < copy.Length; i++)
            {
                int r = rand.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }

            for (int i = 0; i < trainRows; i++)
                trainData[i] = copy[i];
            for (int i = 0; i < testRows; i++)
                testData[i] = copy[i + trainRows];
        }
        public void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }
    }
}

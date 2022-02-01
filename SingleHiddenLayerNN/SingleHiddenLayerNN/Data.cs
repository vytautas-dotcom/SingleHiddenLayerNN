using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleHiddenLayerNN
{
    class Data
    {
        public double[][] rawData = new double[30][];
        public Data()
        {
            rawData[0] = new double[] { 1, 0, 1.4, 0.3, 1, 0, 0 };
            rawData[1] = new double[] { 0, 1, 4.9, 1.5, 0, 1, 0 };
            rawData[2] = new double[] { -1, -1, 5.6, 1.8, 0, 0, 1 };
            rawData[3] = new double[] { -1, -1, 6.1, 2.5, 0, 0, 1 };
            rawData[4] = new double[] { 1, 0, 1.3, 0.2, 1, 0, 0 };
            rawData[5] = new double[] { 0, 1, 1.4, 0.2, 1, 0, 0 };
            rawData[6] = new double[] { 1, 0, 6.6, 2.1, 0, 0, 1 };
            rawData[7] = new double[] { 0, 1, 3.3, 1.0, 0, 1, 0 };
            rawData[8] = new double[] { -1, -1, 1.7, 0.4, 1, 0, 0 };
            rawData[9] = new double[] { 0, 1, 1.5, 0.1, 0, 1, 1 };
            rawData[10] = new double[] { 0, 1, 1.4, 0.2, 1, 0, 0 };
            rawData[11] = new double[] { 0, 1, 4.5, 1.5, 0, 1, 0 };
            rawData[12] = new double[] { 1, 0, 1.4, 0.2, 1, 0, 0 };
            rawData[13] = new double[] { -1, -1, 5.1, 1.9, 0, 0, 1 };
            rawData[14] = new double[] { 1, 0, 6.0, 2.5, 0, 0, 1 };
            rawData[15] = new double[] { 1, 0, 3.9, 1.4, 0, 1, 0 };
            rawData[16] = new double[] { 0, 1, 4.7, 1.4, 0, 1, 0 };
            rawData[17] = new double[] { -1, -1, 4.6, 1.5, 0, 1, 0 };
            rawData[18] = new double[] { -1, -1, 4.5, 1.7, 0, 0, 1 };
            rawData[19] = new double[] { 0, 1, 4.5, 1.3, 0, 1, 0 };
            rawData[20] = new double[] { 1, 0, 1.5, 0.2, 1, 0, 0 };
            rawData[21] = new double[] { 0, 1, 5.8, 2.2, 0, 0, 1 };
            rawData[22] = new double[] { 0, 1, 4.0, 1.3, 0, 1, 0 };
            rawData[23] = new double[] { -1, -1, 5.8, 1.8, 0, 0, 1 };
            rawData[24] = new double[] { 1, 0, 1.5, 0.2, 1, 0, 0 };
            rawData[25] = new double[] { -1, -1, 5.9, 2.1, 0, 0, 1 };
            rawData[26] = new double[] { 0, 1, 1.4, 0.2, 1, 0, 0 };
            rawData[27] = new double[] { 0, 1, 4.7, 1.6, 0, 1, 0 };
            rawData[28] = new double[] { 1, 0, 4.6, 1.3, 0, 1, 0 };
            rawData[29] = new double[] { 1, 0, 6.3, 1.8, 0, 0, 1 };
        }

        public void Info()
        {
            Console.WriteLine("Goal is to predict species from color, petal length, width \n");
            Console.WriteLine("Raw data looks like: \n");
            Console.WriteLine("blue, 1.4, 0.3, setosa");
            Console.WriteLine("pink, 4.9, 1.5, versicolor");
            Console.WriteLine("teal, 5.6, 1.8, virginica \n");
            Console.WriteLine("The independent variable color values are encoded using 1-of-(N-1) effects encoding.");
            Console.WriteLine("Where: ");
            Console.WriteLine("Blue - [ 1.0,  0.0]");
            Console.WriteLine("Pink - [ 0.0,  1.0]");
            Console.WriteLine("Teal - [-1.0, -1.0]");
            Console.WriteLine("The species values are encoded using 1-of-N encoding.");
            Console.WriteLine("Where: ");
            Console.WriteLine("Setosa - "+"[1.0, 0.0, 0.0]".PadLeft(20));
            Console.WriteLine("Veresicolor - "+"[0.0, 1.0, 0.0]".PadLeft(14));
            Console.WriteLine("Virginica - "+"[0.0, 0.0, 1.0]".PadLeft(17));
        }
    }
}

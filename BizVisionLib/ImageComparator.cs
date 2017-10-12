using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Util;

namespace BizVisionLib
{
    public static class ImageComparator
    {

        public static double comparePixel(Image<Bgr, byte> img1, Image<Bgr, byte> img2, double tanhScale = 0.01f)
        {
            Size midSize = new Size((img1.Width + img2.Width) / 2, (img1.Height + img2.Height) / 2);
            img1 = img1.Resize(midSize.Width, midSize.Height, Emgu.CV.CvEnum.Inter.Linear);
            img2 = img2.Resize(midSize.Width, midSize.Height, Emgu.CV.CvEnum.Inter.Linear);
            Mat diffMat = new Mat();
            CvInvoke.AbsDiff(img1, img2, diffMat);
            MCvScalar diffSum = CvInvoke.Sum(diffMat);
            double x = (diffSum.V0 + diffSum.V1 + diffSum.V2) / 3.0 / (midSize.Width * midSize.Height);
            return 1 - Math.Tanh(tanhScale * x);
        }


        public static double compareGradient(Image<Gray, byte> img1, Image<Gray, byte> img2, int density = 4, int histSize = 9)
        {
            Size midSize = new Size((img1.Width + img2.Width) / 2, (img1.Height + img2.Height) / 2);
            img1 = img1.Resize(midSize.Width, midSize.Height, Emgu.CV.CvEnum.Inter.Linear);
            img2 = img2.Resize(midSize.Width, midSize.Height, Emgu.CV.CvEnum.Inter.Linear);
            CvInvoke.Blur(img1, img1, new Size(3, 3), new Point(-1, -1));
            CvInvoke.Blur(img2, img2, new Size(3, 3), new Point(-1, -1));
            Mat des1 = computeHogDes(img1, density, histSize);
            CvInvoke.Normalize(des1, des1);
            Mat des2 = computeHogDes(img2, density, histSize);
            CvInvoke.Normalize(des2, des2);
            double dis = CvInvoke.Norm(des1, des2, Emgu.CV.CvEnum.NormType.L2);
            return 1 - dis < 0 ? 0 : 1 - dis;
        }


        private static Mat computeHogDes(Image<Gray, byte> img, int density = 4, int histSize = 9)
        {
            if (img.Height % density != 0 || img.Width % density != 0)
                img = img.Resize(img.Width - img.Width % density, img.Height - img.Height % density, Emgu.CV.CvEnum.Inter.Linear);
            HOGDescriptor hogd = new HOGDescriptor(img.Size, img.Size, new Size(1, 1),
                new Size(img.Size.Width / density, img.Size.Height / density));
            float[] result = hogd.Compute(img);
            Mat ret = new Mat(result.Length, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            ret.SetTo(result);
            return ret;
        }

    }
}

using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Util;
using System.Collections;

namespace BizVisionLib
{
    public class BulbManager
    {

        public enum IndexMode
        {
            ByRow, ByCol
        }

        /// <summary>
        /// Use gauss mask to sepearte background and evaluate intensity ratio.
        /// </summary>
        /// <param name="image">image of interst</param>
        /// <param name="prospectiveLampSize">size of gauss mask</param>
        /// <param name="sigmoidScale">sigmoid scale of </param>
        /// <param name="sigmaScale">standard deviation of gauss mask</param>
        /// <param name="co">correlation coefficent of gauss mask</param>
        /// <returns>0=not lit, 1=lit</returns>
        public static double isLit(Image<Bgr, byte> image, Size prospectiveLampSize,
             double bias = -10, double sigmoidScale = 0.2, double sigmaScale = 0.5, double co = 0.0)
        {
            prospectiveLampSize = new Size(3, 3);
            SizeF sig = new SizeF((float)(prospectiveLampSize.Width * sigmaScale), 
                (float)(prospectiveLampSize.Height * sigmaScale));
            Image<Gray, float> lightMask = new Image<Gray, float>(prospectiveLampSize);
            Image<Gray, float> backMask = new Image<Gray, float>(image.Size);
            for (int y = 0; y < lightMask.Size.Height; y++)
            {
                for (int x = 0; x < lightMask.Size.Width; x++)
                {
                    double weight = normalDistribution(x, y, prospectiveLampSize.Width / 2,
                        prospectiveLampSize.Height / 2, sig.Width, sig.Height, co);
                    lightMask[y, x] = new Gray(weight);
                    Console.Write(weight + "\t");
                    //backMask[y, x] = new Gray(1 - lightMask[y, x].Intensity);
                }
                Console.WriteLine();
            }
            CvInvoke.Imshow("mask", lightMask);
            CvInvoke.WaitKey(0);
            Mat temp = new Mat();
            CvInvoke.CvtColor(image, temp, Emgu.CV.CvEnum.ColorConversion.Bgr2Hsv);
            Image<Gray, float> lightImg = temp.ToImage<Hsv, float>()[2];
            Image<Gray, float> backImg = temp.ToImage<Hsv, float>()[2];
            CvInvoke.Multiply(lightImg, lightMask, lightImg);
            CvInvoke.Multiply(backImg, backMask, backImg);
            double lightVal = CvInvoke.Sum(lightImg).V0 / CvInvoke.Sum(lightMask).V0;
            double backVal = CvInvoke.Sum(backImg).V0 / CvInvoke.Sum(backMask).V0; ;
            lightImg = lightImg.Mul(1 / 255.0);
            backImg = backImg.Mul(1 / 255.0);
            //CvInvoke.Imshow("lightImg", lightImg);
            //CvInvoke.Imshow("backImg", backImg);
            //Console.WriteLine(lightVal);
            //Console.WriteLine(backVal);
            //CvInvoke.WaitKey(0);
            return 1 / (1 + Math.Exp(-sigmoidScale * (lightVal - backVal + bias)));
        }

        /// <summary>
        /// deprecated. Use threshold to separate background. Performs poor while all black.
        /// </summary>
        public static bool isLit_deprecated(Image<Bgr, byte> image, double validDistance = 0.2, int grayThreshold = -1)
        {
            CvInvoke.CvtColor(image, image, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            Image<Gray, byte> mask = new Image<Gray, byte>(image.Size);
            if (grayThreshold < 0)
            {
                double threshold = CvInvoke.Threshold(image, mask, 0, 255, Emgu.CV.CvEnum.ThresholdType.Otsu);
                Console.WriteLine(threshold);
            }
            else
                CvInvoke.Threshold(image, mask, grayThreshold, 255, Emgu.CV.CvEnum.ThresholdType.Binary);


            List<RectangleF> regions = new List<RectangleF>();
            CvInvoke.Imshow("mask", mask);

            for (int y = 0; y < mask.Height; y++)
            {
                for (int x = 0; x < mask.Width; x++)
                {
                    if (mask[y, x].Intensity != 128)
                    {
                        float val = (float)mask[y, x].Intensity;
                        Mat ptsSet = hfsMarker(mask, x, y, (byte)mask[y, x].Intensity, (byte)128);
                        Mat meanX = new Mat(), meanY = new Mat();
                        Mat stdX = new Mat(), stdY = new Mat();
                        CvInvoke.MeanStdDev(ptsSet.Col(0), meanX, stdX);
                        CvInvoke.MeanStdDev(ptsSet.Col(1), meanY, stdY);
                        float px = (float)(meanX.ToImage<Gray, double>()[0, 0].Intensity);
                        float py = (float)(meanY.ToImage<Gray, double>()[0, 0].Intensity);
                        float stdx = (float)(stdX.ToImage<Gray, double>()[0, 0].Intensity);
                        float stdy = (float)(stdY.ToImage<Gray, double>()[0, 0].Intensity);
                        regions.Add(new RectangleF(px, py, stdx + stdy, val));
                    }
                }
            }

            Console.WriteLine(regions.Count);
            if (regions.Count <= 1)
                return false;
            PointF center = new PointF(mask.Width / 2.0f, mask.Height / 2.0f);
            regions.Sort((RectangleF r1, RectangleF r2) =>
            {
                if (l2Distance(new PointF(r1.X, r1.Y), center) < l2Distance(new PointF(r2.X, r2.Y), center))
                    return -1;
                else
                    return 1;
            });

            foreach (RectangleF rf in regions)
            {
                Console.WriteLine(l2Distance(new PointF(rf.X, rf.Y), center) + ", " + rf.Width + "  - - " + rf.Height);
            }
            if (l2Distance(new PointF(regions[1].X, regions[1].Y), center) > (image.Height + image.Width) * validDistance / 2.0)
                return false;
            if (regions[0].Width < regions[1].Width)
                return regions[0].Height > 128 ? true : false;
            else
                return regions[1].Height > 128 ? true : false;

        }

        /// <summary>
        /// Use kmeans to group markers
        /// </summary>
        /// <param name="markerLocations">marker set</param>
        /// <param name="count">row or col count</param>
        /// <param name="mode">ByRow or ByCol</param>
        /// <returns></returns>
        public static Point[][] indexMarker(List<Point> markerLocations, int count, IndexMode mode)
        {
            Point[][] markerIndices;
            Mat ptsMat = new Mat();
            foreach (Point p in markerLocations)
            {
                Mat t = new Mat(1, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                if (mode == IndexMode.ByRow)
                    t.SetTo<float>(new float[] { p.Y });
                else if (mode == IndexMode.ByCol)
                    t.SetTo<float>(new float[] { p.X });
                ptsMat.PushBack(t);
            }
            Mat centerMat = new Mat(), labelMat = new Mat();
            CvInvoke.Kmeans(ptsMat, count, labelMat, new MCvTermCriteria(100, 0.01), 10, Emgu.CV.CvEnum.KMeansInitType.PPCenters, centerMat);
            Image<Gray, int> labelImg = labelMat.ToImage<Gray, int>();
            markerIndices = new Point[count][];
            for (int i = 0; i < count; i++)
            {
                List<Point> tempList = new List<Point>();
                for (int p = 0; p < labelMat.Rows; p++)
                {
                    if (i == (int)(labelImg[p, 0].Intensity))
                        tempList.Add(markerLocations[p]);
                }
                markerIndices[i] = tempList.ToArray();
            }
            Array.Sort(markerIndices, (Point[] ps1, Point[] ps2) =>
            {
                if (mode == IndexMode.ByRow)
                    return ps1[0].Y - ps2[0].Y;
                else if (mode == IndexMode.ByCol)
                    return ps1[0].X - ps2[0].X;
                return 0;
            });


            for (int i = 0; i < markerIndices.Length; i++)
                Array.Sort(markerIndices[i], (Point p1, Point p2) =>
                    {
                        if (mode == IndexMode.ByRow)
                            return p1.X - p2.X;
                        else if (mode == IndexMode.ByCol)
                            return p1.Y - p2.Y;
                        return 0;
                    });
            return markerIndices;
        }


        /// <summary>
        /// find markers in hsv range
        /// </summary>
        /// <param name="image"></param>
        /// <param name="lowRange">hsv low range</param>
        /// <param name="highRange">hsv high range</param>
        /// <param name="erodeTimes">morphology filter</param>
        /// <returns></returns>
        public static List<Point> findMarkers(Image<Bgr, byte> image, Hsv lowRange, Hsv highRange, int erodeTimes = 5)
        {
            List<Point> markerLocations = new List<Point>();
            Image<Hsv, byte> hsv = new Image<Hsv, byte>(image.Size);
            CvInvoke.CvtColor(image, hsv, Emgu.CV.CvEnum.ColorConversion.Bgr2Hsv);
            Image<Gray, byte> mask = hsv.InRange(lowRange, highRange);
            //CvInvoke.Imshow("masked", mask);
            Mat ele = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(mask, mask, Emgu.CV.CvEnum.MorphOp.Erode, ele, new Point(-1, -1), erodeTimes, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            CvInvoke.MorphologyEx(mask, mask, Emgu.CV.CvEnum.MorphOp.Dilate, ele, new Point(-1, -1), erodeTimes, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            //CvInvoke.Imshow("filtered", mask);

            for (int y = 0; y < mask.Height; y++)
            {
                for (int x = 0; x < mask.Width; x++)
                {
                    if (mask[y, x].Intensity != 0)
                    {
                        Mat ptsSet = hfsMarker(mask, x, y);
                        double px = CvInvoke.Mean(ptsSet.Col(0)).V0;
                        double py = CvInvoke.Mean(ptsSet.Col(1)).V0;
                        markerLocations.Add(new Point((int)px, (int)py));
                    }
                }
            }
            return markerLocations;
        }
        private static Mat hfsMarker(Image<Gray, byte> mask, int initX, int initY, byte val = 255, byte valMask = 0)
        {
            Mat set = new Mat(0, 2, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            Queue<Point> queue = new Queue<Point>();
            queue.Enqueue(new Point(initX, initY));
            while (queue.Count != 0)
            {
                Point p = queue.Dequeue();
                if (p.X < 0 || p.Y < 0 || p.X >= mask.Width || p.Y >= mask.Height)
                    continue;
                if ((int)mask[p.Y, p.X].Intensity == val)
                {
                    Mat t = new Mat(1, 2, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                    t.SetTo<float>(new float[] { p.X, p.Y });
                    set.PushBack(t);
                    mask.Data[p.Y, p.X, 0] = valMask;
                    queue.Enqueue(new Point(p.X - 1, p.Y));
                    queue.Enqueue(new Point(p.X + 1, p.Y));
                    queue.Enqueue(new Point(p.X, p.Y - 1));
                    queue.Enqueue(new Point(p.X, p.Y + 1));
                }
            }
            return set;
        }


        private static double normalDistribution(double x, double y, double ux, double uy, double sigx, double sigy, double co = 0, bool normalize = true)
        {
            double p1 = 1.0 / (2 * Math.PI * sigx * sigy * Math.Sqrt(1 - co * co));
            double p2 = Math.Pow(x - ux, 2) / (sigx * sigx) + Math.Pow(y - uy, 2) / (sigy * sigy);
            double p3 = (-0.5 / (1 - co * co)) * (p2 - (2 * co * (x - ux) * (y - uy)) / (sigx * sigy));
            if (normalize)
                return normalDistribution(0, 0, 0, 0, sigx, sigy, co, false) / p1 * Math.Exp(p3);
            else
                return p1 * Math.Exp(p3);
        }
        private static double l2Distance(PointF p1, PointF p2)
        {
            return Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y));
        }

    }
}

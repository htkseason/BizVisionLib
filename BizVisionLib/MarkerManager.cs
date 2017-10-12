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
using System.Diagnostics;

namespace BizVisionLib
{
    public class MarkerManager
    {
        public struct HomoTrans
        {
            public HomoTrans(Size resolution, Mat homo)
            {
                this.resolution = resolution;
                this.homo = homo;
            }
            public bool isEmpty()
            {
                if (resolution == null || homo == null)
                    return true;
                return false;
            }
            public Size resolution;
            public Mat homo;
        }

        public struct BulbPoint
        {
            public BulbPoint(Point p, Size resolution, byte h, int ft)
            {
                this.location = p;
                this.locationPercent = new PointF(p.X / resolution.Width, p.Y / resolution.Height);
                this.hue = h;
                this.flickerTimes = ft;
            }
            public Point location;
            public PointF locationPercent;
            public byte hue;
            public int flickerTimes;
            public override string ToString()
            {
                return location.ToString() + ", hue=" + hue + ", flicker=" + flickerTimes;
            }
        }

        public static Hsv redLowRange = new Hsv(160, 50, 50);
        public static Hsv redHighRange = new Hsv(20, 256, 256);
        public static Hsv greenLowRange = new Hsv(40, 50, 50);
        public static Hsv greenHighRange = new Hsv(80, 256, 256);
        public static Hsv blueLowRange = new Hsv(100, 50, 50);
        public static Hsv blueHighRange = new Hsv(140, 256, 256);

        public static int ERODE_ADDTIONAL_RETRIEVE_COUNT = 2;

        public static Point[] findMarkers(Image<Bgr, byte> image, Hsv lowRange, Hsv highRange,
            int erodeTimes = 2, bool debug = false)
        {
            List<Point> markerLocations = new List<Point>();
            Image<Hsv, byte> hsv = new Image<Hsv, byte>(image.Size);
            CvInvoke.CvtColor(image, hsv, Emgu.CV.CvEnum.ColorConversion.Bgr2Hsv);
            Image<Gray, byte> mask = hsv.InRange(lowRange, highRange);

            Mat ele = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(mask, mask, Emgu.CV.CvEnum.MorphOp.Erode, ele, new Point(-1, -1), erodeTimes, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            //CvInvoke.MorphologyEx(mask, mask, Emgu.CV.CvEnum.MorphOp.Dilate + ERODE_ADDTIONAL_RETRIEVE_COUNT, ele, new Point(-1, -1), erodeTimes, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            if (debug)
                CvInvoke.Imshow("mask", mask);

            for (int y = 0; y < mask.Height; y++)
                for (int x = 0; x < mask.Width; x++)
                    if (mask.Data[y, x, 0] != 0)
                    {
                        Mat ptsSet = new Mat();
                        hfsMarker(mask, x, y, ptsSet);
                        double px = CvInvoke.Mean(ptsSet.Col(0)).V0;
                        double py = CvInvoke.Mean(ptsSet.Col(1)).V0;
                        markerLocations.Add(new Point((int)px, (int)py));
                    }

            return markerLocations.ToArray();
        }

        public static HomoTrans getHomography(Image<Bgr, byte> image, Point[] pointsList,
            Size resolution, float zoomX = 1f, float zoomY = 1f, int offsetX = 0, int offsetY = 0)
        {
            if (pointsList.Length != 4)
                return new HomoTrans();
            Array.Sort<Point>(pointsList, (Point p1, Point p2) =>
            {
                return p1.Y - p2.Y;
            });
            Point[] pts = new Point[4];
            pts[0] = pointsList[0].X < pointsList[1].X ? pointsList[0] : pointsList[1];
            pts[1] = pointsList[0].X > pointsList[1].X ? pointsList[0] : pointsList[1];
            pts[2] = pointsList[2].X > pointsList[3].X ? pointsList[2] : pointsList[3];
            pts[3] = pointsList[2].X < pointsList[3].X ? pointsList[2] : pointsList[3];

            if (resolution.IsEmpty)
                resolution = new Size((int)(l2Distance(pts[0], pts[1]) + l2Distance(pts[2], pts[3])) / 2,
                    (int)(l2Distance(pts[0], pts[3]) + l2Distance(pts[1], pts[2])) / 2);

            Mat src = new Mat(new Size(2, 4), Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            src.SetTo(new float[] { 0, 0, resolution.Width, 0, 
                resolution.Width, resolution.Height, 0, resolution.Height });
            Mat dst = new Mat(new Size(2, 4), Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            dst.SetTo(new float[]{pts[0].X,pts[0].Y, pts[1].X,pts[1].Y,
                pts[2].X,pts[2].Y,pts[3].X,pts[3].Y,});

            Mat homo = new Mat();
            CvInvoke.FindHomography(src, dst, homo);

            Mat zoomMat = new Mat(new Size(3, 3), Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            zoomMat.SetTo(new float[]
                {1, 0, (1-zoomX)*0.5f*resolution.Width + offsetX,
                0, 1, (1-zoomY)*0.5f*resolution.Height + offsetY, 
                0, 0, 1});
            Image<Gray, float> homoZoomed = homo.ToImage<Gray, float>();
            CvInvoke.Gemm(homoZoomed, zoomMat, 1, new Mat(), 0, homoZoomed);
            Size zoomedResolution = new Size((int)(resolution.Width * zoomX),
            (int)(resolution.Height * zoomY));
            return new HomoTrans(zoomedResolution, homoZoomed.Mat);
        }

        public static Image<Bgr, byte> getAlignedImage(Image<Bgr, byte> image,
            HomoTrans ht)
        {
            Image<Bgr, byte> result = new Image<Bgr, byte>(ht.resolution);
            CvInvoke.WarpPerspective(image, result, ht.homo, ht.resolution, Emgu.CV.CvEnum.Inter.Linear,
                Emgu.CV.CvEnum.Warp.InverseMap, Emgu.CV.CvEnum.BorderType.Constant, new MCvScalar(0, 0, 0, 0));
            return result;
        }


        public static BulbPoint[] findBulb(Image<Bgr, byte> diffImg, int threshold,
            int erodeTimes = 2, bool debug = false)
        {
            CvInvoke.CvtColor(diffImg, diffImg, Emgu.CV.CvEnum.ColorConversion.Bgr2Hsv);
            Image<Gray, byte> diffImgValue = diffImg[2].Clone();

            for (int y = 0; y < diffImg.Height; y++)
                for (int x = 0; x < diffImg.Width; x++)
                    if (diffImgValue.Data[y, x, 0] < threshold)
                        diffImgValue.Data[y, x, 0] = 0;

            Mat ele = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(diffImgValue, diffImgValue,
                Emgu.CV.CvEnum.MorphOp.Erode, ele, new Point(-1, -1), erodeTimes, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            CvInvoke.MorphologyEx(diffImgValue, diffImgValue,
                Emgu.CV.CvEnum.MorphOp.Dilate, ele, new Point(-1, -1), erodeTimes + ERODE_ADDTIONAL_RETRIEVE_COUNT, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            if (debug)
                CvInvoke.Imshow("diffImgValue" + diffImg.GetHashCode(), diffImgValue);

            List<BulbPoint> result = new List<BulbPoint>();
            for (int y = 0; y < diffImg.Height; y++)
                for (int x = 0; x < diffImg.Width; x++)
                    if (diffImgValue.Data[y, x, 0] != 0)
                    {
                        Mat ptsSet = new Mat();
                        double totalIntensity = hfsMarker(diffImgValue, x, y, ptsSet, 0, 0);
                        int px = (int)(CvInvoke.Mean(ptsSet.Col(0)).V0);
                        int py = (int)(CvInvoke.Mean(ptsSet.Col(1)).V0);
                        result.Add(new BulbPoint(new Point(px, py), diffImg.Size,
                            (byte)(diffImg[py, px].Blue), 0));
                    }
            return result.ToArray();
        }

        public static BulbPoint[] findFlicker(VideoCapture vc, HomoTrans homo, int durationMs,
             int leastFliker = 2, int threshold = 80, int erodeTimes = 2, bool debug = false)
        {
            Mat ele = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
            int startTime = System.Environment.TickCount;
            Mat pic1 = new Mat();
            Mat pic2 = new Mat();
            vc.Read(pic1);
            vc.Read(pic2);
            Mat diffPicAcc = new Mat();
            do
            {
                Image<Bgr, byte> diffImg =
                    MarkerManager.getAlignedImage(pic1.ToImage<Bgr, byte>(), homo).AbsDiff(
                    MarkerManager.getAlignedImage(pic2.ToImage<Bgr, byte>(), homo));
                CvInvoke.CvtColor(diffImg, diffImg, Emgu.CV.CvEnum.ColorConversion.Bgr2Hsv);
                Image<Gray, byte> diffImgValue = diffImg[2].Clone();
                for (int y = 0; y < diffImg.Height; y++)
                    for (int x = 0; x < diffImg.Width; x++)
                        if (diffImgValue.Data[y, x, 0] < threshold)
                            diffImgValue.Data[y, x, 0] = 0;
                        else
                            diffImgValue.Data[y, x, 0] = 1;

                if (diffPicAcc.IsEmpty)
                {
                    diffPicAcc.Create(diffImgValue.Rows, diffImgValue.Cols, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
                    diffPicAcc.SetTo(new MCvScalar(0));
                }
                CvInvoke.Add(diffImgValue, diffPicAcc, diffPicAcc);
                Mat t = pic1;
                pic1 = pic2;
                pic2 = t;
                vc.Read(pic2);
            } while (System.Environment.TickCount - startTime < durationMs);

            Image<Gray, byte> diffPicAccI = diffPicAcc.ToImage<Gray, byte>();
            Image<Gray, byte> diffPicAccDebug = null;
            if (debug)
                diffPicAccDebug = diffPicAcc.ToImage<Gray, byte>();
            for (int y = 0; y < diffPicAccI.Height; y++)
                for (int x = 0; x < diffPicAccI.Width; x++)
                    if (diffPicAccI.Data[y, x, 0] < leastFliker)
                        diffPicAccI.Data[y, x, 0] = 0;
                    else if (debug)
                        diffPicAccDebug.Data[y, x, 0] = 255;
            if (debug)
                CvInvoke.Imshow("diffImgAcc", diffPicAccDebug);

            CvInvoke.MorphologyEx(diffPicAccI, diffPicAccI,
                Emgu.CV.CvEnum.MorphOp.Erode, ele, new Point(-1, -1), erodeTimes, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));
            CvInvoke.MorphologyEx(diffPicAccI, diffPicAccI,
                Emgu.CV.CvEnum.MorphOp.Dilate, ele, new Point(-1, -1), erodeTimes + ERODE_ADDTIONAL_RETRIEVE_COUNT, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(0));

            List<BulbPoint> result = new List<BulbPoint>();
            for (int y = 0; y < diffPicAccI.Height; y++)
                for (int x = 0; x < diffPicAccI.Width; x++)
                    if (diffPicAccI.Data[y, x, 0] != 0)
                    {
                        Mat ptsSet = new Mat();
                        double totalIntensity = hfsMarker(diffPicAccI, x, y, ptsSet, 0, 0);
                        int px = (int)(CvInvoke.Mean(ptsSet.Col(0)).V0);
                        int py = (int)(CvInvoke.Mean(ptsSet.Col(1)).V0);
                        int fTimes = (int)Math.Round(totalIntensity / ptsSet.Rows);
                        result.Add(new BulbPoint(new Point(px, py), homo.resolution, 0, fTimes));
                    }
            return result.ToArray();
        }



        private static double hfsMarker(Image<Gray, byte> mask, int initX, int initY, Mat outputPtsSet,
            int val = 255, byte valMask = 0)
        {
            if (outputPtsSet != null)
                outputPtsSet.Create(0, 2, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            Queue<Point> queue = new Queue<Point>();
            queue.Enqueue(new Point(initX, initY));
            double totalVal = 0;
            while (queue.Count != 0)
            {
                Point p = queue.Dequeue();
                if (p.X < 0 || p.Y < 0 || p.X >= mask.Width || p.Y >= mask.Height)
                    continue;

                if ((val < 0 && mask.Data[p.Y, p.X, 0] != -val) ||
                    (val > 0 && mask.Data[p.Y, p.X, 0] == val) ||
                    (val == 0 && mask.Data[p.Y, p.X, 0] != valMask))
                {
                    Mat t = new Mat(1, 2, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                    t.SetTo<float>(new float[] { p.X, p.Y });
                    if (outputPtsSet != null)
                        outputPtsSet.PushBack(t);
                    totalVal += mask.Data[p.Y, p.X, 0];
                    mask.Data[p.Y, p.X, 0] = valMask;
                    queue.Enqueue(new Point(p.X - 1, p.Y));
                    queue.Enqueue(new Point(p.X + 1, p.Y));
                    queue.Enqueue(new Point(p.X, p.Y - 1));
                    queue.Enqueue(new Point(p.X, p.Y + 1));
                }
            }
            return totalVal;
        }

        private static double l2Distance(Point p1, Point p2)
        {
            return Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y));
        }

    }
}

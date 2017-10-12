using BizVisionLib;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.OCR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace BizVisionLib
{
    public class EntryWraper
    {


        /// <summary>
        /// calculate homography transformation struct
        /// </summary>
        /// <param name="originImagePath">input image</param>
        /// <param name="markerLowRange">marker detection low hue range</param>
        /// <param name="markerHighRange">marker detection high hue range</param>
        /// <param name="resolution">
        /// expected resolution of the markered rectangle,
        /// while using Size.Empty, adaptive size will applied (not precise in lenght-width ratio)</param>
        /// <param name="zoomX">
        /// homography zoom ratio, smaller value means draw closer to the plane. 
        /// 1.0 is default. Must greater than 0.</param>
        /// <param name="offsetX">homography offset X in pixel</param>
        /// <param name="offsetY">homography offset Y in pixel</param>
        /// <param name="markerErodeTimes">marker erode filter times</param>
        /// <param name="debug">if true, will show intermediate debug image</param>
        /// <returns>
        /// if four and only four markers are found, a HomoTrans struct will be returned.
        /// Otherwise, an empty homotrans struct will be returned.
        /// Set debug=true to adjust input parameters.
        /// </returns>
        public static MarkerManager.HomoTrans getHomo(string originImagePath,
            Hsv markerLowRange, Hsv markerHighRange, Size resolution,
            float zoomX = 1.0f, float zoomY = 1.0f, int offsetX = 0, int offsetY = 0,
            int markerErodeTimes = 2,
            bool debug = false)
        {
            Image<Bgr, byte> img = new Image<Bgr, byte>(originImagePath);
            MarkerManager mm = new MarkerManager();
            Point[] pts = MarkerManager.findMarkers(img,
                markerLowRange, markerHighRange, markerErodeTimes, debug);
            MarkerManager.HomoTrans homo = MarkerManager.getHomography(
                img, pts, resolution, zoomX, zoomY, offsetX, offsetY);
            return homo;
        }

        /// <summary>
        /// align an image and save the aligned image to file
        /// </summary>
        /// <param name="originImagePath">input image</param>
        /// <param name="alignedImageSavePath">output image path</param>
        /// <param name="homo">homography, using getHomo() to obtain one</param>
        /// <param name="debug">if true, will show intermediate debug image</param>
        public static void getAlignedImage(string originImagePath, string alignedImageSavePath,
            MarkerManager.HomoTrans homo, bool debug = false)
        {
            Image<Bgr, byte> img = new Image<Bgr, byte>(originImagePath);
            Image<Bgr, byte> alignedImg = MarkerManager.getAlignedImage(img, homo);
            alignedImg.Save(alignedImageSavePath);
            if (debug)
                CvInvoke.Imshow("aligned img", alignedImg);
        }

        /// <summary>
        /// detect bulb from a new image and aligned image get from getAlignment().
        /// </summary>
        /// <param name="alignedImgPath">derive from getAlignment()</param>
        /// <param name="homo">derive from getAlignment()</param>
        /// <param name="compareImgPath">new image to compare with</param>
        /// <param name="threshold">bulb threshold, smaller value are more sensitive to noise. 0-255</param>
        /// <param name="erodeTimes">erode filter</param>
        /// <param name="debug">if true, will show intermediate debug image</param>
        /// <returns>
        /// MarkerManager.BulbPoint[][] { (BulbPoint[])litBulbs, (BulbPoint[])litoffBulbs }
        /// </returns>
        public static MarkerManager.BulbPoint[][] detectBulb(
            string alignedImgPath, MarkerManager.HomoTrans homo,
            string compareImgPath, int threshold = 80, int erodeTimes = 2,
            bool debug = false)
        {
            Image<Bgr, byte> alignedImg = new Image<Bgr, byte>(alignedImgPath);
            Image<Bgr, byte> compareImg = new Image<Bgr, byte>(compareImgPath);
            compareImg = MarkerManager.getAlignedImage(compareImg, homo);
            MarkerManager.BulbPoint[] lit = MarkerManager.findBulb(
                compareImg.Sub(alignedImg), threshold, erodeTimes, debug);
            MarkerManager.BulbPoint[] litoff = MarkerManager.findBulb(
                alignedImg.Sub(compareImg), threshold, erodeTimes, debug);
            return new MarkerManager.BulbPoint[][] { lit, litoff };
        }

        /// <summary>
        /// detect bulb flicker
        /// </summary>
        /// <param name="cam">camera object, derive from openCamera()</param>
        /// <param name="homo">derive from getAlignment()</param>
        /// <param name="detectionDurationMs">detection duration in msec</param>
        /// <param name="leastFlikerCount">least filker times</param>
        /// <param name="threshold">bulb threshold, smaller value are more sensitive to noise. 0-255</param>
        /// <param name="erodeTimes">erode filter</param>
        /// <param name="debug">if true, will show intermediate debug image</param>
        /// <returns>Points array contains bulb cordinate</returns>
        public static MarkerManager.BulbPoint[] detectFilcker(VideoCapture cam,
            MarkerManager.HomoTrans homo,
              int detectionDurationMs, int leastFlikerCount,
            int threshold = 80, int erodeTimes = 2,
            bool debug = false)
        {
            return MarkerManager.findFlicker(cam, homo, detectionDurationMs, leastFlikerCount,
                threshold, erodeTimes, debug);
        }

        public static void setCamExposure(VideoCapture cam, double exposure)
        {
            cam.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.Exposure, exposure);
        }
        public static double getCamExposure(VideoCapture cam)
        {
            return cam.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Exposure);
        }

        public static VideoCapture openCam(int cameraId)
        {
            return new VideoCapture(cameraId);
        }
        public static void captureAndSave(VideoCapture cam, string savePath)
        {
            Mat img = new Mat();
            cam.Read(img);
            img.Save(savePath);
        }
        public void closeCam(VideoCapture cam)
        {
            cam.Dispose();
            cam = null;
        }


        /// <summary>
        /// compare two video files.
        /// </summary>
        /// <param name="video1">input video path</param>
        /// <param name="tolerateSecond">if the duration difference of two videos is greater than this parameter (in second unit). returns 0.</param>
        /// <param name="samplingInterval">sampling interval in second unit</param>
        /// <param name="hogDensity">Hog cell density</param>
        /// <returns>similarity from zero to one. One refers to completely same</returns>
        public static double compareVideo(string video1, string video2,
            double tolerateSecond = 2, double samplingInterval = 0.5, int hogDensity = 4)
        {
            VideoCapture v1 = new VideoCapture(video1);
            VideoCapture v2 = new VideoCapture(video2);
            VideoComparator.VideoInfo info1 = new VideoComparator.VideoInfo(v1);
            VideoComparator.VideoInfo info2 = new VideoComparator.VideoInfo(v2);
            if (Math.Abs(info1.duration / info1.fps - info2.duration / info2.fps) > tolerateSecond)
                return 0;
            double alignFrameTime;
            return VideoComparator.compareVideo(v1, v2, out alignFrameTime);
        }


        /// <summary>
        /// compare two image files.
        /// </summary>
        /// <param name="image1">input image path</param>
        /// <param name="hogDensity">hog block density</param>
        /// <returns>similarity from zero to one. One refers to completely same</returns>
        public static double compareImage(string image1, string image2, int hogDensity = 4)
        {
            Mat img1 = new Mat(image1);
            Mat img2 = new Mat(image2);
            Mat img1gray = new Mat();
            Mat img2gray = new Mat();
            CvInvoke.CvtColor(img1, img1gray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(img2, img2gray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            double gval = ImageComparator.compareGradient(img1gray.ToImage<Gray, byte>(), img2gray.ToImage<Gray, byte>(), hogDensity, 9);
            double pval = ImageComparator.comparePixel(img1.ToImage<Bgr, byte>(), img2.ToImage<Bgr, byte>(), 0.01);
            return Math.Min(gval, pval);
        }



        public static MarkerManager.BulbPoint[] filterByRoi(MarkerManager.BulbPoint[] pts, Rectangle roi)
        {
            List<MarkerManager.BulbPoint> result = new List<MarkerManager.BulbPoint>();
            foreach (MarkerManager.BulbPoint p in pts)
                if (isInRoi(p.location, roi))
                    result.Add(p);
            return result.ToArray();
        }

        private static bool isInRoi(Point p, Rectangle roi)
        {
            if (p.X < roi.X || p.Y < roi.Y || p.X > (roi.X + roi.Width) || p.Y > (roi.Y + roi.Height))
                return false;
            return true;
        }

    }
}

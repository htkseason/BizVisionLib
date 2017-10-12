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
    public static class VideoComparator
    {
        public struct VideoInfo
        {
            public double fps;
            public int frameCount;
            public int pos;
            public Size frameSize;
            public double duration;
            public VideoInfo(VideoCapture v)
            {
                fps = v.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps);
                frameSize = new Size((int)(v.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth)),
                        (int)(v.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight)));
                frameCount = (int)(v.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameCount));
                pos = (int)(v.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames));
                duration = frameCount / fps;
            }
        }
        public static double compareVideo(VideoCapture v1, VideoCapture v2, out double alignFrameTime, double samplingInterval = 0.5, int hogDensity = 4)
        {
            Mat v1Des = computeVideoDes(v1);
            Mat v2Des = computeVideoDes(v2);
            Image<Gray, float> template, source;
            if (v1Des.Rows < v2Des.Rows)
            {
                template = v1Des.ToImage<Gray, float>();
                source = v2Des.ToImage<Gray, float>();
            }
            else
            {
                template = v2Des.ToImage<Gray, float>();
                source = v1Des.ToImage<Gray, float>();
            }
            double minDis = Double.MaxValue;
            alignFrameTime = -1;
            for (int i = 0; i <= source.Height - template.Height; i += hogDensity * hogDensity * 9)
            {
                source.ROI = new Rectangle(0, i, 1, template.Height);
                double dis = CvInvoke.Norm(template, source, Emgu.CV.CvEnum.NormType.L2);
                if (dis < minDis)
                {
                    minDis = dis;
                    alignFrameTime = i / (hogDensity * hogDensity * 9) * samplingInterval;
                }
                source.ROI = Rectangle.Empty;
            }
            double eva = minDis / CvInvoke.Norm(template, Emgu.CV.CvEnum.NormType.L2);
            return 1 - eva < 0 ? 0 : 1 - eva;
        }
        public static Mat computeVideoDes(VideoCapture v, double samplingInterval = 0.5, int hogDensity = 4)
        {
            Mat result = new Mat();
            VideoInfo info = new VideoInfo(v);
            Mat temp = new Mat();
            for (int i = 0; i < info.frameCount; i += (int)(info.fps * samplingInterval))
            {
                v.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames, i);
                v.Read(temp);
                CvInvoke.Blur(temp, temp, new Size(3, 3), new Point(-1, -1));
                result.PushBack(computeHogDes(temp.ToImage<Gray, byte>(), hogDensity));
                Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
                Console.Write(Math.Round(((double)i / info.frameCount * 100), 2) + "%");
            }
            Console.WriteLine("..Done");
            return result;
        }

        public static Mat computeHogDes(Image<Gray, byte> img, int density = 4, int histSize = 9)
        {
            if (img.Height % density != 0 || img.Width % density != 0)
                img = img.Resize(img.Width - img.Width % density, img.Height - img.Height % density, Emgu.CV.CvEnum.Inter.Linear);
            HOGDescriptor hogd = new HOGDescriptor(img.Size, img.Size, new Size(1, 1),
                new Size(img.Size.Width / density, img.Size.Height / density), 9);
            float[] result = hogd.Compute(img);
            Mat ret = new Mat(result.Length, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            ret.SetTo(result);
            return ret;
        }
    }
}

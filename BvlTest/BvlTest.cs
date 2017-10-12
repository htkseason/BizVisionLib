using BizVisionLib;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.OCR;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV.UI;
using System.Threading;

namespace BvlTest
{
    public partial class BvlTest : Form
    {
        ImageBox ib = new ImageBox();
        VideoCapture cam = null;
        bool camDisposeFlag = false;
        bool camPauseFlag = false;
        MarkerManager.HomoTrans homo;
        Image<Bgr, byte> alignImg;

        public BvlTest()
        {
            InitializeComponent();
            this.Size = new Size(800, 420);
            ib.Bounds = new Rectangle(0, 0, 512, 384);
            this.Controls.Add(ib);
        }


        private void BvlTest_Load(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                Mat cap = new Mat();
                while (true)
                {
                    if (camDisposeFlag)
                    {
                        if (cam != null)
                            cam.Dispose();
                        camDisposeFlag = false;
                        cam = null;
                    }
                    if (cam != null && cam.IsOpened && !camPauseFlag)
                    {
                        cam.Read(cap);

                        ib.Image = cap.Clone();
                    }
                    else
                    {
                        //ib.Image = null;
                        Thread.Sleep(100);
                    }
                }
            }).Start();

        }
        private void button9_Click(object sender, EventArgs e)
        //open camera
        {
            cam = new VideoCapture((int)numericCam.Value);
        }

        private void button3_Click(object sender, EventArgs e)
        //findmarkers
        {
            MarkerManager.findMarkers(((Mat)ib.Image).ToImage<Bgr, byte>(),
                MarkerManager.greenLowRange, MarkerManager.greenHighRange, (int)numMarkerErode.Value, true);
        }

        private void button10_Click(object sender, EventArgs e)
        //close camera
        {
            camDisposeFlag = true;
        }

        private void button11_Click(object sender, EventArgs e)
        //set align
        {
            MarkerManager mm = new MarkerManager();
            Image<Bgr, byte> img = (((Mat)ib.Image).ToImage<Bgr, byte>());
            Point[] pts = MarkerManager.findMarkers(img,
                MarkerManager.greenLowRange, MarkerManager.greenHighRange, (int)numMarkerErode.Value);
            MarkerManager.HomoTrans ht = MarkerManager.getHomography(img, pts, Size.Empty,
                (float)numZoomX.Value, (float)numZoomY.Value, (int)numOffsetX.Value, (int)numOffsetY.Value);
            if (!ht.isEmpty())
            {
                this.homo = ht;
                this.alignImg = MarkerManager.getAlignedImage(img, this.homo);
                CvInvoke.Imshow("aligned img", this.alignImg);
            }
            else
                MessageBox.Show("cannot decide homography, pts count = " + pts.Length);
        }
        private void button4_Click(object sender, EventArgs e)
        //check align
        {
            if (!homo.isEmpty())
            {
                CvInvoke.Imshow("aligned img",
                    MarkerManager.getAlignedImage((((Mat)ib.Image).ToImage<Bgr, byte>()),
                    this.homo));
            }
        }

        private void button1_Click(object sender, EventArgs e)
        // find bulb
        {
            Image<Bgr, byte> img = (((Mat)ib.Image).ToImage<Bgr, byte>());
            img = MarkerManager.getAlignedImage(img, this.homo);
            MarkerManager.BulbPoint[] lit = MarkerManager.findBulb(img.Sub(alignImg), (int)numBulbThreshold.Value,
               (int)numBulbErode.Value, true);
            MarkerManager.BulbPoint[] litoff = MarkerManager.findBulb(alignImg.Sub(img), (int)numBulbThreshold.Value,
                (int)numBulbErode.Value, true);
            Console.WriteLine("lit");
            foreach (MarkerManager.BulbPoint p in lit)
                Console.WriteLine(p);
            Console.WriteLine("litoff:");
            foreach (MarkerManager.BulbPoint p in litoff)
                Console.WriteLine(p);
        }

        private void button2_Click(object sender, EventArgs e)
        // find flicker
        {
            if (homo.isEmpty() || cam == null || !cam.IsOpened)
            {
                MessageBox.Show("initiate first");
                return;
            }
            camPauseFlag = true;
            Thread.Sleep(100);
            MarkerManager.BulbPoint[] pts = MarkerManager.findFlicker(cam, homo,
                (int)((float)numFlickerSecond.Value * 1000),
             (int)numLeastFlicker.Value, (int)numBulbThreshold.Value, (int)numBulbErode.Value, true);
            camPauseFlag = false;
            Console.WriteLine("flicker:");
            foreach (MarkerManager.BulbPoint p in pts)
                Console.WriteLine(p.ToString());
        }


        private void button5_Click(object sender, EventArgs e)
        {
            Console.WriteLine(EntryWraper.compareImage(textBoxCompare1.Text, textBoxCompare2.Text));
        }
        private void button6_Click(object sender, EventArgs e)
        {
            Console.WriteLine(EntryWraper.compareVideo(
                textBoxCompare1.Text, textBoxCompare2.Text,
                2, 0.5));
        }



        private Tesseract.Character[] ocr(Image<Bgr, byte> image, Rectangle roi)
        {
            OcrWraper ocr = new OcrWraper();
            image.ROI = roi;
            Tesseract.Character[] chars = ocr.recognize(image);
            image.ROI = Rectangle.Empty;
            return chars;
        }

        private void button7_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Multiselect = false; ;
            dialog.Filter = "All(*.*)|*.*";
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                textBoxCompare1.Text = dialog.FileName;
                try
                {
                    Mat img = new Mat(dialog.FileName);
                    ib.Image = img.Clone();
                    if ((double)img.Width / img.Height > (double)ib.Width / ib.Height)
                        ib.SetZoomScale((double)ib.Width / img.Width, new Point(0, 0));
                    else
                        ib.SetZoomScale((double)ib.Height / img.Height, new Point(0, 0));
                    ib.Refresh();
                }
                catch (Exception) { }
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Multiselect = false; ;
            dialog.Filter = "All(*.*)|*.*";
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                try
                {
                    textBoxCompare2.Text = dialog.FileName;
                    Mat img = new Mat(dialog.FileName);
                    ib.Image = img.Clone();
                    if ((double)img.Width / img.Height > (double)ib.Width / ib.Height)
                        ib.SetZoomScale((double)ib.Width / img.Width, new Point(0, 0));
                    else
                        ib.SetZoomScale((double)ib.Height / img.Height, new Point(0, 0));
                    ib.Refresh();
                }
                catch (Exception) { }
            }
        }

        private void button12_Click(object sender, EventArgs e)
        {
            EntryWraper.setCamExposure(cam, EntryWraper.getCamExposure(cam) + 1);
            Console.WriteLine(EntryWraper.getCamExposure(cam));
        }

        private void button13_Click(object sender, EventArgs e)
        {
            EntryWraper.setCamExposure(cam, EntryWraper.getCamExposure(cam) - 1);
            Console.WriteLine(EntryWraper.getCamExposure(cam));
        }



    }
}

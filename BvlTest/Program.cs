using BizVisionLib;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.OCR;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;


namespace BvlTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //markerDemo(@"../../testPic/a.jpg");
            //markerDemo(@"../../kb1.jpg", @"../../kb2.jpg", new Size(400, 200), 80);
            //ocrDemo();
            //compareVideoDemo(@"../../video_10-20.avi", @"../../video_origin.avi");
            //cameraDemo();
            //compareImageDemo();
            //Console.ReadLine();
            
            Thread t = new Thread(() =>
             {
                 new BvlTest().Show();
                 Application.Run();
             });
            t.SetApartmentState(ApartmentState.STA);
            t.Start();
            
            Console.ReadLine();
        }

    }
}

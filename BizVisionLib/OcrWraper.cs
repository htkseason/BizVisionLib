using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.OCR;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Util;

namespace BizVisionLib
{
    public class OcrWraper
    {
        private Tesseract ocr;
        public OcrWraper()
        {
            ocr = new Tesseract("", "eng", OcrEngineMode.TesseractLstmCombined);
        }
        public Tesseract.Character[] recognize(IInputArray image)
        {
            ocr.SetImage(image);
            ocr.Recognize();
            return ocr.GetCharacters();
        }
    }
}

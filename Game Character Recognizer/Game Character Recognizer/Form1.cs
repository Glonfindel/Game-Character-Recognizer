using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.IO;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.Util;

namespace Game_Character_Recognizer
{
    public partial class Form1 : Form
    {
        VideoCapture camera;
        Image<Bgr, byte> frame;
        string[] files;
        List<Image<Bgr, byte>> images = new List<Image<Bgr, byte>>();
        bool recognize = true;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                camera = new VideoCapture();
                camera.SetCaptureProperty(CapProp.FrameWidth, 400);
                camera.SetCaptureProperty(CapProp.FrameHeight, 300);
            }
            catch (Exception ex)
            {
                Text = ex.Message;
                return;
            }

            files = Directory.GetFiles(@".\images\", "*.jpg");
            foreach (var file in files)
            {
                images.Add(new Image<Bgr, byte>(new Bitmap(file)));
            }

            Application.Idle += new EventHandler(this.Detect);
        }

        public void Detect(object sender, EventArgs arg)
        {
            try
            {
                frame = camera.QueryFrame().ToImage<Bgr, Byte>();
                if (recognize)
                {
                    int i = 0;
                    foreach (var image in images)
                    {
                        using (Mat modelImage = image.Mat)
                        using (Mat observedImage = frame.Mat)
                        {
                            Draw(modelImage, observedImage, Path.GetFileName(files[i]));
                        }
                        i++;
                    }
                }
            }
            catch (Exception ex)
            {
                Text = ex.Message;
            }
            imageBox.Image = frame;
        }

        static void FindMatch(Mat modelImage, Mat observedImage, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.60;

            homography = null;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();

            using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
            {
                KAZE featureDetector = new KAZE();

                Mat modelDescriptors = new Mat();
                featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);

                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);

                using (Emgu.CV.Flann.LinearIndexParams ip = new Emgu.CV.Flann.LinearIndexParams())
                using (Emgu.CV.Flann.SearchParams sp = new SearchParams())
                using (Emgu.CV.Features2D.DescriptorMatcher matcher = new FlannBasedMatcher(ip, sp))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, matches, mask, 2);
                    }
                }
            }
        }

        public void Draw(Mat modelImage, Mat observedImage, string name)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out modelKeyPoints, out observedKeyPoints, matches, out mask, out homography);

                if (homography != null)
                {
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                        new PointF(rect.Left, rect.Bottom),
                        new PointF(rect.Right, rect.Bottom),
                        new PointF(rect.Right, rect.Top),
                        new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);

                    frame.DrawPolyline(points, true, new Bgr(Color.Red), 2);
                    label1.Text = name.Substring(0, name.LastIndexOf('.'));
                }
            }
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            label1.Text = "";
            label1.Visible = true;
            button1.Visible = false;
            textBox1.Visible = false;
            label2.Visible = false;
            recognize = true;
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            label1.Visible = false;
            button1.Visible = true;
            textBox1.Visible = true;
            label2.Visible = true;
            recognize = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Image<Bgr, byte> temp;
            temp = frame.Resize(240, 180, Emgu.CV.CvEnum.Inter.Cubic);
            temp.Save(@".\images\" + textBox1.Text + ".jpg");
            textBox1.Text = "";
            files = null;
            images = new List<Image<Bgr, byte>>();
            files = Directory.GetFiles(@".\images\", "*.jpg");
            foreach (var file in files)
            {
                images.Add(new Image<Bgr, byte>(new Bitmap(file)));
            }
        }
    }
}
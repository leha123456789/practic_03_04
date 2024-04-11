#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

const int WIDTH = 800;
const int HEIGHT = 600;
const double MIN_REAL = -2.0;
const double MAX_REAL = 1.0;
const double MIN_IMAGINARY = -1.5;
const double MAX_IMAGINARY = 1.5;
const int MAX_ITERATIONS = 1000;

Vec3b getColor(int iterations) 
{
    if (iterations == MAX_ITERATIONS) 
    {
        return Vec3b(0, 0, 0); 
    }
    else 
    {
        double hue = fmod(iterations * 0.02, 1.0); 
        double saturation = 1.0;
        double value = 1.0;
        int bgr[3];
        int p = int(hue * 6);
        double f = hue * 6 - p;
        double q = value * (1 - saturation);
        double t = value * (1 - saturation * f);
        double v = value * (1 - saturation * (1 - f));
        switch (p) 
        {
        case 0:
            bgr[0] = int(value * 255);
            bgr[1] = int(v * 255);
            bgr[2] = int(q * 255);
            break;
        case 1:
            bgr[0] = int(t * 255);
            bgr[1] = int(value * 255);
            bgr[2] = int(q * 255);
            break;
        case 2:
            bgr[0] = int(q * 255);
            bgr[1] = int(value * 255);
            bgr[2] = int(v * 255);
            break;
        case 3:
            bgr[0] = int(q * 255);
            bgr[1] = int(t * 255);
            bgr[2] = int(value * 255);
            break;
        case 4:
            bgr[0] = int(v * 255);
            bgr[1] = int(q * 255);
            bgr[2] = int(value * 255);
            break;
        case 5:
            bgr[0] = int(value * 255);
            bgr[1] = int(q * 255);
            bgr[2] = int(t * 255);
            break;
        }
        return Vec3b(bgr[2], bgr[1], bgr[0]);
    }
}

bool isInMandelbrotSet(double real, double imaginary) 
{
    double cReal = real;
    double cImaginary = imaginary;
    double zReal = 0.0;
    double zImaginary = 0.0;

    for (int i = 0; i < MAX_ITERATIONS; ++i) 
    {
        double zRealTemp = zReal * zReal - zImaginary * zImaginary + cReal;
        double zImaginaryTemp = 2 * zReal * zImaginary + cImaginary;
        zReal = zRealTemp;
        zImaginary = zImaginaryTemp;
        if (zReal * zReal + zImaginary * zImaginary > 4) 
        {
            return false; 
        }
    }
    return true; 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image(HEIGHT, WIDTH, CV_8UC3);

    double real, imaginary;
    bool inMandelbrotSet;

    int startRow = rank * (HEIGHT / size);
    int endRow = (rank + 1) * (HEIGHT / size);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            real = MIN_REAL + j * (MAX_REAL - MIN_REAL) / (WIDTH - 1);
            imaginary = MIN_IMAGINARY + i * (MAX_IMAGINARY - MIN_IMAGINARY) / (HEIGHT - 1);
            inMandelbrotSet = isInMandelbrotSet(real, imaginary);
            Vec3b color;
            if (inMandelbrotSet) {
                color = getColor(MAX_ITERATIONS); 
            }
            else 
            {
                int iterations = 0;
                double zReal = 0.0;
                double zImaginary = 0.0;
                double cReal = real;
                double cImaginary = imaginary;

                for (; iterations < MAX_ITERATIONS; ++iterations) 
                {
                    double zRealTemp = zReal * zReal - zImaginary * zImaginary + cReal;
                    double zImaginaryTemp = 2 * zReal * zImaginary + cImaginary;
                    zReal = zRealTemp;
                    zImaginary = zImaginaryTemp;
                    if (zReal * zReal + zImaginary * zImaginary > 4) 
                    {
                        break; 
                    }
                }
                color = getColor(iterations); 
            }
            image.at<Vec3b>(i, j) = color;
        }
    }
    Mat resultImage(HEIGHT, WIDTH, CV_8UC3);
    MPI_Gather(image.data, image.total() * image.elemSize(), MPI_UNSIGNED_CHAR, resultImage.data, image.total() * image.elemSize(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        imshow("Фрактал", resultImage);
        imwrite("Фрактал.png", resultImage); 
        waitKey(0);
    }
    MPI_Finalize();
    return 0;
}

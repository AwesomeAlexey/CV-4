#include <iostream>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include <string>

using namespace std;
using namespace cv;


static inline vector<string> get_files(const string& directory);
void PrettySpectrum(Mat& fourier);
Mat performDFTConvolve(Mat a, Mat b, Mat c);
Mat make_n_show_magnitude(Mat src, string prefix="", bool pretty=false);
Mat getSpectrum(Mat src, Mat* base = nullptr);
Mat getLowFreq(Mat src);
Mat getHighFreq(Mat src);





int main() {

    vector<vector<int8_t>> filters = {
            {-1, -2, -1, 0, 0, 0, 1, 2, 1},
            {-1, 0, 1, -2, 0, 2, -1, 0, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            {0, 1, 0, 1, -4, 1, 0, 1, 0}
    };

    string directory = "../НужноБольшеФурье";

    auto files = get_files(directory);

    for (const auto& file: files){

        cout << file << endl;
        Mat img = imread(file, IMREAD_GRAYSCALE);
        imshow("Orig", img);
        img.convertTo(img, CV_32FC1);

        for(auto filter:filters) {

            Mat f = {
                    Size(3, 3), CV_8S,
                    filter.data()
            };

            f.convertTo(f, CV_32FC1);

            Mat res;

            res = performDFTConvolve(img, f, res);

            normalize(res, res, 0, 1, NORM_MINMAX);
            imshow("Res", res);
            waitKey(0);
        }

        destroyAllWindows();

        Mat spectrum = getSpectrum(img);
        Mat magnitude = make_n_show_magnitude(spectrum, "");
        PrettySpectrum(spectrum);
        make_n_show_magnitude(spectrum, "Before_");

        Mat low = getLowFreq(spectrum);
        Mat high = getHighFreq(spectrum);

        make_n_show_magnitude(low, "Before_L");
        make_n_show_magnitude(low, "Before_H");

        PrettySpectrum(low);
        PrettySpectrum(high);

        make_n_show_magnitude(low, "After_L");
        make_n_show_magnitude(high, "After_H");

        dft(low, img, DFT_INVERSE | DFT_REAL_OUTPUT);
        imshow("Res_L", img);
        dft(high, img, DFT_INVERSE | DFT_REAL_OUTPUT);
        imshow("Res_H", img);
        waitKey(0);

        destroyAllWindows();
    }


    Mat plate = imread("../license_plate.png", IMREAD_GRAYSCALE);
    imshow("Plate", plate);

    Mat a(plate.clone(), Rect(105, 66, 90, 120));
    imshow("A", a);
    Mat zero(plate.clone(), Rect(220, 40, 90, 150));
    imshow("0", zero);
    Mat six(plate.clone(), Rect(317, 40, 90, 150));
    imshow("6", six);

    Mat symbols[3] = {a, zero, six};

    Mat plate_f = getSpectrum(plate);

    for (auto symbol: symbols){

        Mat temp;
        Mat symbol_f = getSpectrum(symbol, &plate);
        make_n_show_magnitude(plate_f, "Plate", true);
        make_n_show_magnitude(symbol_f, "Symbol", true);
        mulSpectrums(plate_f, symbol_f, temp, 0, true);
        make_n_show_magnitude(temp, "Correlated", true);
        Mat res;
        dft(temp, res, DFT_INVERSE | DFT_REAL_OUTPUT);
        normalize(res, res, 0, 1, NORM_MINMAX);
        imshow("Result", res);
        Point max_loc;
        double max;

        minMaxLoc(res, nullptr, &max, &max_loc, nullptr);
        cout << max << endl;

        threshold(res, res, max - 0.01, 255, CV_8U);

        imshow("Threshold", res);
        waitKey(0);


    }

    waitKey(0);

}




static inline vector<string> get_files(const string& directory){

    vector<string> files;

    for (const auto& entry: filesystem::directory_iterator(directory)){
        files.push_back(entry.path());
    }
    return files;

}


void PrettySpectrum(Mat& fourier){

    int cx = fourier.cols / 2;
    int cy = fourier.rows / 2;

    Mat q0(fourier, Rect(0, 0, cx, cy));
    Mat q1(fourier, Rect(cx, 0, cx, cy));
    Mat q2(fourier, Rect(0, cy, cx, cy));
    Mat q3(fourier, Rect(cx, cy, cx, cy));

    Mat temp;
    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);

}


Mat make_n_show_magnitude(Mat src, string prefix, bool pretty){
    Mat chs[2];

    split(src, chs);


    Mat magn;

    magnitude(chs[0], chs[1], magn);

    magn+= Scalar::all(1);
    log(magn, magn);

    normalize(magn, magn, 0, 1, NormTypes::NORM_MINMAX);

    if(pretty){
        PrettySpectrum(magn);
    }

    imshow(prefix.append(" magnitude"), magn);


    Mat magn_hsv;
    cvtColor(magn, magn_hsv, COLOR_GRAY2RGB);

    cvtColor(magn_hsv, magn_hsv, COLOR_RGB2HSV);

    return magn;
}



Mat performDFTCorrelation(Mat a, Mat b){

    Size dft_size;
    dft_size.width = getOptimalDFTSize(a.cols + b.cols - 1);
    dft_size.height = getOptimalDFTSize(a.rows + b.rows - 1);


    Mat tempA(dft_size, a.type(), Scalar::all(0));
    Mat tempB(dft_size, b.type(), Scalar::all(0));


    Mat roiA(tempA, Rect(0, 0, a.cols, a.rows));
    Mat roiB(tempB, Rect(0, 0, b.cols, b.rows));
    a.copyTo(roiA);
    b.copyTo(roiB);

    dft(tempA, tempA, DFT_COMPLEX_OUTPUT);
    dft(tempB, tempB, DFT_COMPLEX_OUTPUT);

    mulSpectrums(tempA, tempB, tempA, 0, true);

    Mat c = a.clone();

    tempA(Rect(0, 0, c.cols, c.rows)).copyTo(c);
    return c;


}

Mat performDFTConvolve(Mat a, Mat b, Mat c){

    Size dft_size;
    dft_size.width = getOptimalDFTSize(a.cols + b.cols - 1);
    dft_size.height = getOptimalDFTSize(a.rows + b.rows - 1);


    c.create(a.rows, a.cols, a.type());

    Mat tempA(dft_size, a.type(), Scalar::all(0));
    Mat tempB(dft_size, b.type(), Scalar::all(0));


    Mat roiA(tempA, Rect(0, 0, a.cols, a.rows));
    Mat roiB(tempB, Rect(0, 0, b.cols, b.rows));
    a.copyTo(roiA);
    b.copyTo(roiB);

    dft(tempA, tempA, DFT_COMPLEX_OUTPUT);
    dft(tempB, tempB, DFT_COMPLEX_OUTPUT);

    make_n_show_magnitude(tempA, "Original", true);
    make_n_show_magnitude(tempB, "Kernel", true);

    mulSpectrums(tempA, tempB, tempA, 0);

    dft(tempA, tempA, DFT_INVERSE | DFT_REAL_OUTPUT);

    tempA(Rect(0, 0, c.cols, c.rows)).copyTo(c);
    return c;

}



Mat getSpectrum(Mat src, Mat* base){

    Size dft_size;
    if(base == nullptr) {
        dft_size.width = getOptimalDFTSize(src.cols);
        dft_size.height = getOptimalDFTSize(src.rows);
    } else{
        dft_size.width = getOptimalDFTSize(base->cols);
        dft_size.height = getOptimalDFTSize(base->rows);
    }
    Mat temp(dft_size, src.type(), Scalar::all(0));

    Mat roi(temp, Rect(0, 0, src.cols, src.rows));
    src.copyTo(roi);
    temp.convertTo(temp, CV_32FC1);
    dft(temp, temp, DFT_COMPLEX_OUTPUT);

    return temp;

}


Mat getLowFreq(Mat src){

    Mat mask(Size(src.cols, src.rows), CV_8U, Scalar::all(255));
    Point center(src.cols/2, src.rows/2);

    circle(mask, center, src.cols/4, Scalar::all(0), -1);

    Mat temp;
    src.copyTo(temp, mask);

    return temp;

}

Mat getHighFreq(Mat src){
    Mat mask(Size(src.cols, src.rows), CV_8U, Scalar::all(0));
    Point center(src.cols/2, src.rows/2);

    circle(mask, center, src.cols/2, Scalar::all(255), -1);

    Mat temp;
    src.copyTo(temp, mask);

    return temp;

}
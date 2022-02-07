#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

void go_find_eggs(string link, int cnt);

double r2 = sqrt(2);

// 이미지를 불러올 폴더와 내보낼 폴더를 생성하여 경로를 적어야 함.
String input_path = "C:\\Users\\newyo\\Documents\\OpenCV_goEgg_detecting\\egg_image_input";
String output_path_txt = "C:\\Users\\newyo\\Documents\\OpenCV_goEgg_detecting\\egg_image_output";

int main()
{
	String path(input_path + "\\*");

	vector<String> str;

	glob(path, str, false);

	for (int cnt = 0; cnt < str.size(); cnt++) {
		go_find_eggs(str[cnt], cnt);
	}

	return 0;
}

void go_find_eggs(string link, int cnt)
{
	Mat src = imread(link, IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat blurred;
	blur(src, blurred, Size(3, 3));
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	// 바둑알 검출
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 1, 150, 30, 10, 12);

	// 현재 차례
	if (circles.size() % 2 == 0) {
		cout << "바둑판의 돌의 개수 : " << circles.size() << ", 흑돌 차례" << endl;
	}
	else {
		cout << "바둑판의 돌의 개수 : " << circles.size() << ", 백돌 차례" << endl;
	}

	// 바둑알의 평균 반지름
	float z = 0;
	for (Vec3f c : circles) {
		z = z + c.val[2];
	}
	float average_r = z / circles.size();

	vector<Vec3f> egg(circles.size(), Vec3f(3, 0));
	int eggcount = 0;

	// 검출한 원 정보 egg벡터로 저장하기.
	for (Vec3f c : circles) {
		int dotx = cvRound(c[0]) - cvRound(c[2]) / r2;
		int doty = cvRound(c[1]) - cvRound(c[2]) / r2;
		int radius = cvRound(c[2]);

		// 바둑알
		Rect in_egg_rect(dotx, doty, 2 * radius / r2, 2 * radius / r2);

		int size_egg_rect = in_egg_rect.height * in_egg_rect.width;

		Point clean(cvRound(c[0]), cvRound(c[1]) - radius / r2);
		drawMarker(dst, clean, Scalar(0, 0, 255), 0, 2, 2, 8);
		int clean_gray = blurred.at<uchar>(clean);

		if (clean_gray < 128) {
			egg[eggcount].val[0] = c[0] - c[2];
			egg[eggcount].val[1] = c[1] - c[2];
			egg[eggcount].val[2] = 0;
			rectangle(dst, in_egg_rect, Scalar(0, 0, 255), 0.6);
		}
		else {
			egg[eggcount].val[0] = c[0] - c[2];
			egg[eggcount].val[1] = c[1] - c[2];
			egg[eggcount].val[2] = 1; // 백돌에 1 부여.
			rectangle(dst, in_egg_rect, Scalar(255, 0, 0), 0.6);
		}
		eggcount++;
	}

	// 바둑판 검출
	Mat gray = src.clone();

	Mat bin;
	threshold(gray, bin, 200, 255, THRESH_BINARY_INV | THRESH_OTSU);

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	
	// 근사화된 외곽선 저장할 벡터.
	vector<Point> approx;
	for (vector<Point>& pts : contours) {
		if (contourArea(pts) < 400)
			continue;
		// 근사 점 마다 벡터에 저장.
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);
	}
	//for (int i = 0; i < 4; i++) {
	//	cout << approx[i]; // 좌상, 좌하, 우하, 우상 순서임.
	//}
	int fx = (int)approx[3].x;
	int ly = (int)approx[3].y;
	// cout << fx << " " << ly << endl;

	// 19*19 바둑판 점 저장 벡터. 위에서 검출한 rect 정보로.
	// 같은 int 값인데 fx를 쓰면 blank_area출력과 cout이 잘 작동을 안함.
	// fx, ly 말고 33, 467을 쓰면 잘 됨; 같은 int 값인데. 밑의 int 값은 마우스이벤트로 구한 네 꼭짓점 좌표 값으로 구한 것.
	vector<Vec2i> dot(19 * 19, Vec2i(2, 0));
	int dotcount = 0;
	int a = (467 - 33) / 18;
	for (int i = 33; i < 467; i += a) {
		for (int j = 33; j < 467; j += a) {
			dot[dotcount].val[0] = j;
			dot[dotcount].val[1] = i;
			dotcount++;
		}
	}

	// 텍스트 파일 출력 준비
	cnt++;
	String result_text_name = output_path_txt + "\\img" + to_string(cnt) + "_text";
	ofstream fout(result_text_name);

	int dotnumber = 0;
	for (Vec2i d : dot) {
		dotnumber++;
		cout << ".";
		fout << ".";
		Rect blank_area(d.val[0] - a, d.val[1] - a, a * 2, a * 2);
		//rectangle(dst, blank_area, Scalar(0, 0, 255), 1, 8);
		for (Vec3f e : egg) {
			Rect egg_area(e.val[0] - e.val[2], e.val[1] - e.val[2], average_r * 2 + 2.5, average_r * 2 + 2.5);
			rectangle(dst, egg_area, Scalar(0, 0, 0), 1, 8);
			Rect k = blank_area & egg_area;
			if (k == egg_area) {
				if (e.val[2] == 0) {
					cout << "\bb";
					fout << "\bb";
				}
				else if (e.val[2] == 1) {
					cout << "\bw";
					fout << "\bw";
				}
			}
		}
		if ((dotnumber % 19) == 0) {
			cout << "\n";
			fout << "\n";
		}
	}
	fout.close();

	imshow("dst", dst);

	String result_image_name = output_path_txt + "\\img" + to_string(cnt) + ".jpg";

	vector<int> params;
	params.push_back(IMWRITE_JPEG_QUALITY);
	params.push_back(95);
	imwrite(result_image_name, dst, params);

	waitKey();
	destroyAllWindows();
}
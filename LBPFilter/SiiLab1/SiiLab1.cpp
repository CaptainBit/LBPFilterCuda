#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include "stdafx.h" 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/imgproc/imgproc.hpp>
#include "AxisCommunication.h"

using namespace cv;
using namespace std;

//Fonction pour appeler le kernel à partir du cpu
extern "C" cudaError_t ApplyLBPFilter(Mat *ptMatA, Mat *ptMatR);


int main()
{
	//Variable qui va détenir le code d'erreur(s'il y en a) sinon sera cudaSucces
	cudaError_t status;

	//Image lena
	Mat lenasrc = imread("lenna256x256.bmp", 0);
	Mat lenaLBP = imread("lenna256x256.bmp", 0);
	
	//Appeler le Kernel
	status = ApplyLBPFilter(&lenasrc, &lenaLBP);

	//Vérifier s'il y a une erreur
	if (status != cudaSuccess) 
	{
		return 0;
	}
	imshow("Lena src", lenasrc);
	imshow("Lena Test", lenaLBP);

	//image legumes
	Mat legumessrc = imread("Legumes.jpg", 0);
	Mat legumesLBP = imread("Legumes.jpg", 0);

	//Appeler le Kernel
	status = ApplyLBPFilter(&legumessrc, &legumesLBP);

	//Vérifier s'il y a une erreur
	if (status != cudaSuccess)
	{
		return 0;
	}
	imshow("Legumes src", legumessrc);
	imshow("Legumes Test", legumesLBP);

	//image penguins
	Mat penguinssrc = imread("Penguins.jpg", 0);
	Mat penguinsLBP = imread("Penguins.jpg", 0);

	//Appeler le Kernel
	status = ApplyLBPFilter(&penguinssrc, &penguinsLBP);

	//Vérifier s'il y a une erreur
	if (status != cudaSuccess)
	{
		return 0;
	}

	imshow("Penguins src", penguinssrc);
	imshow("Penguins Test", penguinsLBP);

	//image player
	Mat joueursrc = imread("Player.jpg", 0);
	Mat joueurLBP = imread("Player.jpg", 0);

	//Appeler le Kernel
	status = ApplyLBPFilter(&joueursrc, &joueurLBP);

	//Vérifier s'il y a une erreur
	if (status != cudaSuccess)
	{
		return 0;
	}

	imshow("Player src", joueursrc);
	imshow("Player Test", joueurLBP);

	waitKey(0);
	return 0;
}

//Cuda file 
#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/imgproc/imgproc.hpp>  



using namespace cv; 

__global__ void Kernel_LBPFilter(unsigned char *MatA, unsigned char *MatR, int rows, int cols);

__device__  int Mask(int pi, int po);

int iDivUp(int a, int b);

extern "C"	cudaError_t ApplyLBPFilter(Mat *ptMatA, Mat *ptMatR)
{
	cudaError_t status;

	//pointeurs des matrices 
	uchar *MatA, *MatR;

	//Dimension de la grid et des blocs 
	dim3 nbreThreadsParBlock(32, 32);
	dim3 nbreBloc(iDivUp(ptMatA->cols, 32), iDivUp(ptMatA->rows, 32));

	//Calculer l'espace nécessaire dans la mémoire du gpu
	int memSize = ptMatA->rows * ptMatA->step1();

	//Allouer espace pour le gpu 
	cudaMalloc((void **)&MatA, memSize);
	cudaMalloc((void **)&MatR, memSize);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		goto Error;
	}
	//Envoyer matrice dans la mémoire du gpu 
	cudaMemcpy(MatA, ptMatA->data, memSize, cudaMemcpyHostToDevice);
	//Check status
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		goto Error;
	}

	Kernel_LBPFilter <<<nbreBloc, nbreThreadsParBlock >>>(MatA, MatR, ptMatA->step1(), ptMatA->rows);

	//Check status
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		goto Error;
	}
	//Wait the Kernel to be done
	cudaDeviceSynchronize();
	//Retourner la matrice résultante 
	cudaMemcpy(ptMatR->data, MatR, memSize, cudaMemcpyDeviceToHost);
	//Check status
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		goto Error;
	}
	//Libérer espace mémoire dans le gpu
	cudaFree(MatA);
	cudaFree(MatR);
	return status;
Error:
	cudaFree(MatA);
	cudaFree(MatR);
	return status;
}
 
__global__ void Kernel_LBPFilter(unsigned char *MatA, unsigned char *MatR, int rows, int cols)
{

	//X et Y dans la matrice 
	int ImgNumColonne = (blockIdx.x  * blockDim.x) + threadIdx.x;
	int ImgNumLigne = (blockIdx.y * blockDim.y) + threadIdx.y;

	//Ne depasse pas l'accès de la matrice
	if ((ImgNumColonne < (rows)-1) && (ImgNumLigne < (cols)-2))
	{
		
		//Total addition
		int total = 0;
		//Exposant
		int exp = 1;
		//Indice du Po
		int x5 = ((ImgNumLigne + 1) * rows) + ((ImgNumColonne)* 3) + 1;
		//Valeur du po (initial)
		int po = MatA[x5];

		//Pour chaque ligne(3)
		for (int iL = 1; iL <= 3; iL++) 
		{
			//Pour chaque colonne(3)
			for (int iC = 1; iC <= 3; iC++) 
			{
				//(Valeur de mon indice)
				int xpi = ((ImgNumLigne + iL - 1) * rows) + ((ImgNumColonne * 3) + (iC - 1));

				//Pour ne pas calculer le po
				if (xpi != x5) 
				{
					//Masque
					int mpi = Mask(MatA[xpi], po);

					mpi = mpi * exp;
					total = total + mpi;
					exp = exp * 2;
				}
				
			}
		}
		//Mettre résultat dans la matrice
		MatR[x5] = total;
	}

}

//Retourne 0 ou 1 (Pi = 0 si Pi<0 sinon =1)
__device__  int Mask(int pi, int po) 
{
	if (pi >= po) 
	{
		return 1;
	}
	else 
	{
		return 0;
	}
}

int iDivUp(int a, int b) // Round a / b to nearest higher integer value

{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


 
 
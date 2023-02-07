#include <iostream>
#include <cmath>
#include <typeinfo>
#include <time.h>
#include <opencv2/opencv.hpp>

// command line args for opencv
// -lopencv_core -lopencv_imgcodecs

// main command: nvcc main.cu -lopencv_core -lopencv_imgcodecs && ./a.out img1.jpg

using namespace cv;
using namespace std;

struct RLE
{
  int value;
  int count;
};

int pairingCPU(int x, int y)
{
  return (x>=y) ? (x*x + x + y) : (y*y + x);
}

RLE unpairingCPU(int z)
{
  struct RLE res;
  int b = (int)sqrt(z);
  int a = z - b*b;
  if(a<b)
  {
    res.value=a;
    res.count=b;
  }
  else
  {
    res.value=b;
    res.count=a-b;
  }
  return res;
}

__global__ void pairingGPU(const RLE *rle, int *res, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    int x = rle[i].value;
    int y = rle[i].count;
    res[i] = (x>=y) ? (x*x + x + y) : (y*y + x);
  }
}

__global__ void unpairingGPU(const int *compressed, RLE *decompressed, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    int z = *(compressed+i);
    int b = (int)sqrt((float)z);
    int a = z - b*b;

    RLE rle;
    if(a<b)
    {
      rle.value = a;
      rle.count = b;
    }
    else
    {
      rle.value = b;
      rle.count = a-b;
    }
    *(decompressed+i) = rle;
  }
}

void printVectorInt(vector<int> v, int n)
{
  for(int i=0; i<n; i++)
    cout << v[i] << " ";
  cout << endl;
}

void printVectorRLE(vector<RLE> v, int n)
{
  for(int i=0; i<n; i++)
    cout << "[" << v[i].value << "," << v[i].count << "] ";
  cout << endl;
}

void printArrayInt(int *v, int n)
{
  for(int i=0; i<n; i++)
    cout << *(v+i) << " ";
  cout << endl;
}

void printArrayRLE(RLE *v, int n)
{
  for(int i=0; i<n; i++)
  {
    RLE x = *(v+i);
    cout << "[" << x.value << "," << x.count << "] ";
  }
  cout << endl;
}

int main(int argc, char *argv[])
{
  clock_t begin, end;
  double time_spent1, time_spent2;

  // Load the image using OpenCV
  Mat image = imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Failed to load image!" << endl;
    return 1;
  }

  vector<RLE> compressed;
  int count = 0;
  int prevPixel = static_cast<int>(image.at<unsigned char>(0,0));
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      int currPixel = static_cast<int>(image.at<unsigned char>(row,col));
      if(currPixel == prevPixel)
        count++;
      else {
        struct RLE rle;
        rle.value = currPixel;
        rle.count = count;

        compressed.push_back(rle);

        prevPixel = currPixel;
        count = 1;
      }
    }
  }


  // CPU
  cout << endl << "```````CPU```````" << endl;
  cout << "after compression:            ";
  printVectorRLE(compressed, 5);

  // further compression by using pairing function
  begin = clock();
  vector<int> enhCompressed;
  for(auto item : compressed)
    enhCompressed.push_back(pairingCPU(item.value, item.count));
  end = clock();
  time_spent1 = (double)(end - begin) / CLOCKS_PER_SEC;
  cout << "after enhanced compression:   ";
  printVectorInt(enhCompressed, 5);
  
  // do decompression
  begin = clock();
  vector<RLE> enhDecompressed;
  for(auto item : enhCompressed)
    enhDecompressed.push_back(unpairingCPU(item));
  end = clock();
  time_spent2 = (double)(end - begin) / CLOCKS_PER_SEC;
  cout << "after enhanced decompression: ";
  printVectorRLE(enhDecompressed, 5);
  
  // compare the decompressed and original versions, should be 100% same
  // i.e. compare compressed and enhDecompressed vectors
  int cnt1=0;
  for(int i=0; i<enhDecompressed.size(); i++)
    if(enhDecompressed[i].value==compressed[i].value && enhDecompressed[i].count==compressed[i].count)
      cnt1++;
  cout << endl << "Percentage of same elements: " << ((cnt1)/(float)(compressed.size()))*100.0 << endl;

  cout << fixed;
  cout.precision(8);
  cout << "Time for compression:   " << time_spent1 << " seconds" << endl;
  cout << "Time for decompression: " << time_spent2 << " seconds" << endl;
  cout.precision(0);


  // GPU
  cout << endl << "```````GPU```````" << endl;
  cout << "after compression:            ";
  printVectorRLE(compressed, 5);

  // GPU datastructures initialization and stuff
  int size1 = compressed.size();
  int threadsPerBlock = 8;
  int blocksPerGrid = size1;
  // 3 vectors in host   (CPU) - compressedC, enhCompressedC, enhDecompressedC
  RLE *compressedC      = (RLE *)malloc(size1*sizeof(RLE));
  int *enhCompressedC   = (int *)malloc(size1*sizeof(int));
  RLE *enhDecompressedC = (RLE *)malloc(size1*sizeof(RLE));
  // initialize them
  for(int i=0; i<size1; i++)
  {
    compressedC[i].value = compressed[i].value;
    compressedC[i].count = compressed[i].count;
    
    enhCompressedC[i] = 0;
    enhDecompressedC[i].value = 0;
    enhDecompressedC[i].count = 0;
  }
  // 3 vectors in device (GPU) - compressedG, enhCompressedG, enhDecompressedG
  RLE *compressedG      = NULL;
  int *enhCompressedG   = NULL;
  RLE *enhDecompressedG = NULL;
  cudaMalloc((void **)&compressedG     , size1*sizeof(RLE));
  cudaMalloc((void **)&enhCompressedG  , size1*sizeof(int));
  cudaMalloc((void **)&enhDecompressedG, size1*sizeof(RLE));
  // copy host vector to device vector
  cudaMemcpy(compressedG, compressedC, size1*sizeof(RLE), cudaMemcpyHostToDevice);

  // further compression by using pairing function
  begin = clock();
  pairingGPU<<<blocksPerGrid, threadsPerBlock>>>(compressedG, enhCompressedG, size1);
  end = clock();
  time_spent1 = (double)(end - begin) / CLOCKS_PER_SEC;
  // copy back to host and print it
  cudaMemcpy(enhCompressedC, enhCompressedG, size1*sizeof(int), cudaMemcpyDeviceToHost);
  cout << "after enhanced compression:   ";
  printArrayInt(enhCompressedC, 5);

  // do decomposition
  begin = clock();
  unpairingGPU<<<blocksPerGrid, threadsPerBlock>>>(enhCompressedG, enhDecompressedG, size1);
  end = clock();
  time_spent2 = (double)(end - begin) / CLOCKS_PER_SEC;
  // copy back to host and print it
  cudaMemcpy(enhDecompressedC, enhDecompressedG, size1*sizeof(RLE), cudaMemcpyDeviceToHost);
  cout << "after enhanced decompression: ";
  printArrayRLE(enhDecompressedC, 5);

  // compare the decompressed and original versions, should be 100% same
  // i.e. compare compressedC and enhDecompressedC vectors
  cnt1=0;
  for(int i=0; i<size1; i++)
  {
    RLE rle1, rle2;
    rle1 = *(enhDecompressedC+i);
    rle2 = *(compressedC+i);
    if(rle1.value==rle2.value && rle1.count==rle2.count)
      cnt1++;
  }
  cout << endl << "Percentage of same elements: " << ((cnt1)/(float)(size1))*100.0 << endl;

  cout << fixed;
  cout.precision(8);
  cout << "Time for compression:   " << time_spent1 << " seconds" << endl;
  cout << "Time for decompression: " << time_spent2 << " seconds" << endl;
  cout.precision(4);


  // Some comparisions
  cout << endl << "```````STATISTICS```````" << endl;
  int originalSize = image.rows * image.cols * 1; // 8-bit image (8 bits = 1 byte)
  int compressedSize = (1+sizeof(int)) * compressed.size(); // 1 btye for color value + 1 int for count
  int enhCompressedSize = sizeof(int) * compressed.size(); // 1 int
  cout << "original size of image:   " << originalSize      << " bytes" << endl;
  cout << "compressed size:          " << compressedSize    << " bytes" << endl;
  cout << "enhanced compressed size: " << enhCompressedSize << " bytes" << endl;
  cout << endl;
  cout << "compression ratio (org|compr):       " << ((originalSize)/(float)(compressedSize)) << endl;
  cout << "compression ratio (compr|enh_compr): " << ((compressedSize)/(float)(enhCompressedSize)) << endl;

  cnt1 = 0;
  for(auto item : compressed)
    if (item.count>1)
      cnt1++;
  cout << "Found " << cnt1 << " repetitive sequences, image size = " << image.rows << " X " << image.cols
       << " (" << (image.rows*image.cols) << " pixels)" << endl;

  return 0;
}


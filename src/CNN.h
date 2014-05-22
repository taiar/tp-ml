#ifndef CNN_H_
#define CNN_H_

#include <math.h>

#define UINT unsigned int

const UINT g_cImageSize = 28;
const UINT g_cVectorSize = 29;
const UINT g_cOutputSize = 10;
const UINT g_cCountHessianSample = 500;
const UINT g_cCountTrainingSample = 60000;
const UINT g_cCountTestingSample = 10000;


#define RGB_TO_BGRQUAD(r,g,b) (RGB((b),(g),(r)))

#define GAUSSIAN_FIELD_SIZE (21)

#define RANDOM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 ) //in [-1, 1] range
#define RANDOM_ZERO_ONE     ( (double) rand() / (RAND_MAX + 1) )    //in [0, 1) range

const double dMaxScaling = 15.0; //like 20 for 20%
const double dMaxRotation = 15.0;  // like 20.0 for 20 degrees
const double dElasticSigma = 4.0;//8.0;  // higher numbers are more smooth and less distorted; Simard uses 4.0
const double dElasticScaling = 0.34;//0.5;  // higher numbers amplify the distortions; Simard uses 34 (sic, maybe 0.34 ??)
const double dMicronLimitParameter = 0.10;  // since we divide by this, update can never be more than 10x current eta

const double PI = 2.0 * acos(0.0);


#include <math.h>
#include <cstdlib>
#include <cstdio>

typedef unsigned char uchar;

#define INPUT_LAYER   0
#define CONVOLUTIONAL 1
#define FULLY_CONNECTED 2

#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))// derivative of the sigmoid as a function of the sigmoid's output

class Layer;

class FeatureMap {
public:
  double bias, dErr_wrtb, diagHessianBias;
  double *value, *dError;
  double **kernel, **diagHessian, **dErr_wrtw;
  int m_nFeatureMapPrev;

  Layer *pLayer;

  void Construct( );
  void Delete();
  void Clear();
  void ClearDError();
  void ClearDiagHessian();
  void ClearDErrWRTW();

  double Convolute(double *input, int size, int r0, int c0, double *weight, int kernel_size);
  void Calculate(double *valueFeatureMapPrev, int idxFeatureMapPrev );
  void BackPropagate(double *valueFeatureMapPrev, int idxFeatureMapPrev, double *dErrorFeatureMapPrev, int dOrder );
};

class Layer {
public:
  int m_type;
  int m_SamplingFactor;

  Layer *pLayerPrev;

  int m_nFeatureMap;
  int m_FeatureSize;
  int m_KernelSize;

  FeatureMap* m_FeatureMap;

  void ClearAll() {
    for(int i=0; i<m_nFeatureMap; i++) {
      m_FeatureMap[i].Clear();
      m_FeatureMap[i].ClearDError();
      m_FeatureMap[i].ClearDErrWRTW();
      m_FeatureMap[i].ClearDiagHessian();
    }
  }

  void print(char *fileName, double *array, int size) {
    FILE *f = fopen(fileName, "w");
    for(int i=0; i<size; i++)
      fprintf(f, "%lg\n", array[i]);
  }

  void Calculate();
  void BackPropagate(int dOrder, double etaLearningRate);

  void Construct(int type, int nFeatureMap, int FeatureSize, int KernelSize, int SamplingFactor);
  void Delete();
};

class CCNN {
public:
  CCNN(void);
  CCNN(int numConvolutionalLayers, int numHiddenLayers,
    int * convFeatureMaps, int * convKernelSizes,
    int * convStepSize, int * hiddenLayerUnits, size_t classCnt,
    size_t sqrtInput);
  ~CCNN(void);

  Layer *m_Layer;
  int m_nLayer;

  void ConstructNN();
  void ConstructNN(
    int numConvolutionalLayers,     // The number of convolutional layers (placed immediatly after the input layer)
    int numHiddenLayers,            // The number of hidden, MLP layers at the end
    int * convFeatureMaps,          // The number of feature maps in each convolutional layer
    int * convKernelSizes,          // The size of the side of the kernel square
    int * convStepSize,             // Inverse of the pool size ratio; A step size of 3 means that 1/3rd of the neurons in the previous layer get used
    int * hiddenLayerUnits,         // The number of units in each hidden layer
    size_t classCnt,                 // The number of classes in the classification
    size_t sqrtInput                // The square root of the input (assumes input is a square)
  );
  void DeleteNN();

  void LoadWeights(char *FileName);
  void LoadWeightsRandom();
  void SaveWeights(char *FileName);
  int Calculate(double *input, double *output);
  void BackPropagate(double *desiredOutput, double eta);
  void CalculateHessian( );
};

#endif
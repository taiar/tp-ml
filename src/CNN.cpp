#include "CNN.h"
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <algorithm> // For max
#include <float.h>

CCNN::CCNN(void) {
  ConstructNN();
}

CCNN::CCNN(
  int numConvolutionalLayers,     // The number of convolutional layers (placed immediatly after the input layer)
  int numHiddenLayers,            // The number of hidden, MLP layers at the end
  int * convFeatureMaps,          // The number of feature maps in each convolutional layer
  int * convKernelSizes,          // The size of the side of the kernel square
  int * convStepSize,             // Inverse of the pool size ratio; A step size of 3 means that 1/3rd of the neurons in the previous layer get used
  int * hiddenLayerUnits,         // The number of units in each hidden layer
  size_t classCnt,                // The number of classes in the classification
  size_t sqrtInput                // The square root of the input (assumes input is a square)
){
  ConstructNN(numConvolutionalLayers, numHiddenLayers, convFeatureMaps, convKernelSizes, convStepSize, hiddenLayerUnits, classCnt, sqrtInput);
}

CCNN::~CCNN(void) {
  DeleteNN();
}

void CCNN::ConstructNN(
  int numConvolutionalLayers,     // The number of convolutional layers (placed immediatly after the input layer)
  int numHiddenLayers,            // The number of hidden, MLP layers at the end
  int * convFeatureMaps,          // The number of feature maps in each convolutional layer
  int * convKernelSizes,          // The size of the side of the kernel square
  int * convStepSize,             // Inverse of the pool size ratio; A step size of 3 means that 1/3rd of the neurons in the previous layer get used
  int * hiddenLayerUnits,         // The number of units in each hidden layer
  size_t classCnt,                // The number of classes in the classification
  size_t sqrtInput                // The square root of the input (assumes input is a square)
){
  // Generate Layers
  m_nLayer = 1 + numConvolutionalLayers + numHiddenLayers + 1; // Input + conv + MLP + output
  m_Layer = new Layer[m_nLayer];

  // Chain layers together
  m_Layer[0].pLayerPrev = NULL;
  for (int i = 1; i < m_nLayer; i++) m_Layer[i].pLayerPrev = &m_Layer[i - 1];

  // Initialize input layer
  m_Layer[0].Construct(INPUT_LAYER, 1, sqrtInput, 0, 0);

  // Initialize convolution layers
  for (int i = 0; i < numConvolutionalLayers; i++){
    m_Layer[i + 1].Construct(
      CONVOLUTIONAL,
      convFeatureMaps[i],
      (m_Layer[i].m_FeatureSize - convKernelSizes[i]) / convStepSize[i] + 1,
      convKernelSizes[i],
      convStepSize[i]
    );
  }

  // Initialize MLP hidden-layers
  for (int i = 0; i < numHiddenLayers; i++){
    m_Layer[i + numConvolutionalLayers + 1].Construct(
      FULLY_CONNECTED,
      hiddenLayerUnits[i],
      1,
      m_Layer[i + numConvolutionalLayers].m_FeatureSize,
      1
    );
  }

  // Initialize MLP output-layer -- Decays into a perceptron if 0 hidden layers were set
  m_Layer[m_nLayer - 1].Construct(
    FULLY_CONNECTED,
    (int)classCnt,
    1,
    numHiddenLayers ? 1 : m_Layer[m_nLayer - 2].m_FeatureSize,
    1
  );
}

void CCNN::ConstructNN(){
  int featureMapCnt[] = { 13, 5 };
  int kernelSizes[] = { 5, 5 };
  int stepSizes[] = { 2, 2 };
  int hiddenCnt[] = { 100 };
  ConstructNN(2, 1, featureMapCnt, kernelSizes, stepSizes, hiddenCnt, 10, 29);
}

void CCNN::DeleteNN() {
  for(int i=0; i<m_nLayer; i++) m_Layer[i].Delete();
}

void CCNN::LoadWeightsRandom() {
  int i, j, k, m;

  srand((unsigned)time(0));

  for ( i=1; i<m_nLayer; i++ ) {
    for( j=0; j<m_Layer[i].m_nFeatureMap; j++ ) {
      m_Layer[i].m_FeatureMap[j].bias = 0.05 * RANDOM_PLUS_MINUS_ONE;

      for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
        for(m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++)
          m_Layer[i].m_FeatureMap[j].kernel[k][m] = 0.05 * RANDOM_PLUS_MINUS_ONE;
    }
  }
}

void CCNN::LoadWeights(char *FileName) {
  int i, j, k, m;

  FILE *f;
  if((f = fopen(FileName, "r")) == NULL) return;

  for ( i=1; i<m_nLayer; i++ ) {
    for( j=0; j<m_Layer[i].m_nFeatureMap; j++ ) {
      fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].bias);

      for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
        for(m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++)
          fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel[k][m]);
    }
  }
  fclose(f);
}

void CCNN::SaveWeights(char *FileName) {
  int i, j, k, m;

  FILE *f;
  if((f = fopen(FileName, "w")) == NULL) return;

  for ( i=1; i<m_nLayer; i++ ) {
    for( j=0; j<m_Layer[i].m_nFeatureMap; j++ ) {
      fprintf(f, "%lg ", m_Layer[i].m_FeatureMap[j].bias);

      for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
        for(m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++) {
          fprintf(f, "%lg ", m_Layer[i].m_FeatureMap[j].kernel[k][m]);
        }
    }
  }

  fclose(f);

}

int CCNN::Calculate(double *input, double *output) {
  int i, j;

  //copy input to layer 0
  for(i=0; i<m_Layer[0].m_nFeatureMap; i++)
    for(j=0; j < m_Layer[0].m_FeatureSize * m_Layer[0].m_FeatureSize; j++)
      m_Layer[0].m_FeatureMap[0].value[j] = input[j];

  //forward propagation
  //calculate values of neurons in each layer
  for(i=1; i<m_nLayer; i++) {
    //initialization of feature maps to ZERO
    for(j=0; j<m_Layer[i].m_nFeatureMap; j++) m_Layer[i].m_FeatureMap[j].Clear();

    //forward propagation from layer[i-1] to layer[i]
    m_Layer[i].Calculate();
  }

  //copy last layer values to output
  int maxIdx = 0;
  double maxVal = -DBL_MAX;
  for (i = 0; i < m_Layer[m_nLayer - 1].m_nFeatureMap; i++) {
    double val = m_Layer[m_nLayer - 1].m_FeatureMap[i].value[0];
    if(output) output[i] = m_Layer[m_nLayer - 1].m_FeatureMap[i].value[0];
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }

  return maxIdx;
}

void CCNN::BackPropagate(double *desiredOutput, double eta) {
  int i;

  //derivative of the error in last layer
  //calculated as difference between actual and desired output (eq. 2)
  for(i=0; i<m_Layer[m_nLayer-1].m_nFeatureMap; i++) {
    m_Layer[m_nLayer-1].m_FeatureMap[i].dError[0] =
        m_Layer[m_nLayer-1].m_FeatureMap[i].value[0] - desiredOutput[i];
  }

  double mse=0.0;
  for ( i=0; i<10; i++ )
    mse += m_Layer[m_nLayer-1].m_FeatureMap[i].dError[0] * m_Layer[m_nLayer-1].m_FeatureMap[i].dError[0];

  //backpropagate through rest of the layers
  for(i=m_nLayer-1; i>0; i--)
    m_Layer[i].BackPropagate(1, eta);

  int t=0;
}

void CCNN::CalculateHessian() {
  int i, j, k;

  //2nd derivative of the error wrt Xn in last layer
  //it is always 1
  //Xn is the output after applying SIGMOID

  for(i=0; i<m_Layer[m_nLayer-1].m_nFeatureMap; i++)
    m_Layer[m_nLayer-1].m_FeatureMap[i].dError[0] = 1.0;

  //backpropagate through rest of the layers
  for(i=m_nLayer-1; i>0; i--)
    m_Layer[i].BackPropagate(2, 0);

  //average over the number of samples used
  for(i=1; i<m_nLayer; i++)
    for(j=0; j<m_Layer[i].m_nFeatureMap; j++) {
      m_Layer[i].m_FeatureMap[j].diagHessianBias /= g_cCountHessianSample;
      for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
        for(int m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++)
          m_Layer[i].m_FeatureMap[j].diagHessian[k][m] /= g_cCountHessianSample;
    }
  int t=0;
}

void Layer::Construct(int type, int nFeatureMap, int FeatureSize, int KernelSize,
  int SamplingFactor) {
  m_type = type;
  m_nFeatureMap = nFeatureMap;
  m_FeatureSize = FeatureSize;
  m_KernelSize = KernelSize;
  m_SamplingFactor = SamplingFactor;

  m_FeatureMap = new FeatureMap[ m_nFeatureMap ];

  for(int j=0; j<m_nFeatureMap; j++) {
    m_FeatureMap[j].pLayer = this;
    m_FeatureMap[j].Construct(  );
  }
}

void Layer::Delete() {
  for(int j=0; j<m_nFeatureMap; j++) m_FeatureMap[j].Delete();
}

//forward propagation
void Layer::Calculate() {
  for(int i=0; i<m_nFeatureMap; i++) {

    //initialize feature map to bias
    for(int k=0; k < m_FeatureSize * m_FeatureSize; k++)
        m_FeatureMap[i].value[k] = m_FeatureMap[i].bias;

    //calculate effect of each feature map in previous layer
    //on this feature map in this layer
    for(int j=0; j<pLayerPrev->m_nFeatureMap; j++) {
      m_FeatureMap[i].Calculate(
        pLayerPrev->m_FeatureMap[j].value,  //input feature map
        j                                   //index of input feature map
      );
    }

    //SIGMOD function
    for(int j=0; j < m_FeatureSize * m_FeatureSize; j++)
      m_FeatureMap[i].value[j] = 1.7159 * tanh(0.66666667 * m_FeatureMap[i].value[j]);
  }
}

void Layer::BackPropagate(int dOrder, double etaLearningRate) {
  //find dError (2nd derivative) wrt the actual output Yn of this layer
  //Note that SIGMOID was applied to Yn to get Xn during forward propagation
  //We already have dErr_wrt_dXn calculated in CCNN :: BackPropagate and
  //use the following equation to get dErr_wrt_dYn
  //dErr_wrt_dYn = InverseSIGMOID(Xn)^2 * dErr_wrt_dXn

  for(int i=0; i<m_nFeatureMap; i++) {
    for(int j=0; j < m_FeatureSize * m_FeatureSize; j++) {
      double temp = DSIGMOID(m_FeatureMap[i].value[j]);
      if(dOrder == 2) temp *= temp;
      m_FeatureMap[i].dError[j] = temp * m_FeatureMap[i].dError[j];
    }
  }

  //clear dError wrt weights
  for(int i=0; i<m_nFeatureMap; i++)
    m_FeatureMap[i].ClearDErrWRTW();

  //clear dError wrt Xn in previous layer.
  //This is input to the previous layer for backpropagation
  for(int i=0; i<pLayerPrev->m_nFeatureMap; i++)
    pLayerPrev->m_FeatureMap[i].ClearDError();

  //Backpropagate
  for(int i=0; i<m_nFeatureMap; i++) {
    //derivative of error wrt bias
    for(int j=0; j<m_FeatureSize * m_FeatureSize; j++)
      m_FeatureMap[i].dErr_wrtb += m_FeatureMap[i].dError[j];

    //calculate effect of this feature map on each feature map in the revious layer
    for(int j=0; j<pLayerPrev->m_nFeatureMap; j++) {
      m_FeatureMap[i].BackPropagate(
        pLayerPrev->m_FeatureMap[j].value,    // input feature map
        j,                                    // index of input feature map
        pLayerPrev->m_FeatureMap[j].dError,   // dErr_wrt_Xn for previous layer
        dOrder                                // order of derivative
      );
    }
  }

  //update weights (for backporpagation) or diagonal hessian (for 2nd order backpropagation)
  double epsilon, divisor;

  for(int i=0; i<m_nFeatureMap; i++) {
    if(dOrder == 1) {
      divisor = std::max(0.0, m_FeatureMap[i].diagHessianBias) + dMicronLimitParameter;
      epsilon = etaLearningRate / divisor;
      m_FeatureMap[i].bias -= epsilon * m_FeatureMap[i].dErr_wrtb;
    } else
      m_FeatureMap[i].diagHessianBias += m_FeatureMap[i].dErr_wrtb;

    for(int j=0; j<pLayerPrev->m_nFeatureMap; j++) {
      for(int k=0; k < m_KernelSize * m_KernelSize; k++) {
        if(dOrder == 1) {
          divisor = std::max(0.0, m_FeatureMap[i].kernel[j][k]) + dMicronLimitParameter;
          epsilon = etaLearningRate / divisor;
          m_FeatureMap[i].kernel[j][k] -= epsilon * m_FeatureMap[i].dErr_wrtw[j][k];
        } else
          m_FeatureMap[i].diagHessian[j][k] += m_FeatureMap[i].dErr_wrtw[j][k];
      }
    }
  }
}

void FeatureMap::Construct() {
  if(pLayer->m_type == INPUT_LAYER)
    m_nFeatureMapPrev = 0;
  else
    m_nFeatureMapPrev = pLayer->pLayerPrev->m_nFeatureMap;

  int FeatureSize = pLayer->m_FeatureSize;
  int KernelSize  = pLayer->m_KernelSize;

  //neuron values
  value = new double [ FeatureSize * FeatureSize ];

  //error in neuron values
  dError = new double [ FeatureSize * FeatureSize ];

  //weights kernel
  kernel = new double* [ m_nFeatureMapPrev ];
  for(int i=0; i<m_nFeatureMapPrev; i++) {
    kernel[i] = new double [KernelSize * KernelSize];

    //initialize
    bias = 0.05 * RANDOM_PLUS_MINUS_ONE;
    for(int j=0; j < KernelSize * KernelSize; j++) kernel[i][j] = 0.05 * RANDOM_PLUS_MINUS_ONE;
  }

  //diagHessian
  diagHessian = new double* [ m_nFeatureMapPrev ];
  for(int i=0; i<m_nFeatureMapPrev; i++)
    diagHessian[i] = new double [KernelSize * KernelSize];

  //derivative of error wrt kernel weights
  dErr_wrtw = new double* [ m_nFeatureMapPrev ];
  for(int i=0; i<m_nFeatureMapPrev; i++)
    dErr_wrtw[i] = new double [KernelSize * KernelSize];
}

void FeatureMap::Delete() {
  delete[] value;
  delete[] dError;
  for(int i=0; i<m_nFeatureMapPrev; i++) {
    delete[] kernel[i];
    delete[] dErr_wrtw[i];
    delete[] diagHessian[i];
  }
}

void FeatureMap::Clear() {
  for(int i=0; i < pLayer->m_FeatureSize * pLayer->m_FeatureSize; i++) value[i] = 0.0;
}

void FeatureMap::ClearDError() {
  for(int i=0; i < pLayer->m_FeatureSize * pLayer->m_FeatureSize; i++) dError[i] = 0.0;
}

void FeatureMap::ClearDiagHessian() {
  diagHessianBias = 0;
  for(int i=0; i < m_nFeatureMapPrev; i++)
    for(int j=0; j < pLayer->m_KernelSize * pLayer->m_KernelSize; j++) diagHessian[i][j] = 0.0;
}

void FeatureMap::ClearDErrWRTW() {
  dErr_wrtb = 0;
  for(int i=0; i < m_nFeatureMapPrev; i++)
    for(int j=0; j < pLayer->m_KernelSize * pLayer->m_KernelSize; j++)
      dErr_wrtw[i][j] = 0.0;
}

//calculate effect of a feature map in previous layer on this feature map in this layer
//  valueFeatureMapPrev:  feature map in previous layer
//  idxFeatureMapPrev :   index of feature map in previous layer
void FeatureMap::Calculate(double *valueFeatureMapPrev, int idxFeatureMapPrev ) {
  int isize = pLayer->pLayerPrev->m_FeatureSize; //feature size in previous layer
  int ksize = pLayer->m_KernelSize;
  int step_size = pLayer->m_SamplingFactor;

  int k = 0;

  for(int row0 = 0; row0 <= isize - ksize; row0 += step_size)
    for(int col0 = 0; col0 <= isize - ksize; col0 += step_size)
      value[k++] += Convolute(valueFeatureMapPrev, isize, row0, col0, kernel[idxFeatureMapPrev], ksize);
}

double FeatureMap::Convolute(double *input, int size, int r0, int c0, double *weight,
  int kernel_size) {
  int i, j, k = 0;
  double summ = 0;

  for(i = r0; i < r0 + kernel_size; i++)
    for(j = c0; j < c0 + kernel_size; j++)
      summ += input[i * size + j] * weight[k++];

  return summ;
}

//calculate effect of this feature map on a feature map in previous layer
//note that previous layer is next in backpropagation
//  valueFeatureMapPrev:    feature map in previous layer
//  idxFeatureMapPrev :     index of feature map in previous layer
//  dErrorFeatureMapPrev:   dError wrt neuron values in the FM in prev layer
void FeatureMap::BackPropagate(double *valueFeatureMapPrev, int idxFeatureMapPrev,
  double *dErrorFeatureMapPrev, int dOrder) {

  int isize = pLayer->pLayerPrev->m_FeatureSize;  //size of FM in previous layer
  int ksize = pLayer->m_KernelSize;       //kernel size
  int step_size = pLayer->m_SamplingFactor;   //subsampling factor
  int row0, col0, k = 0;

  for(row0 = 0; row0 <= isize - ksize; row0 += step_size) {
    for(col0 = 0; col0 <= isize - ksize; col0 += step_size) {
      for(int i=0; i<ksize; i++) {
        for(int j=0; j<ksize; j++) {
          //get dError wrt output for feature map in the previous layer
          double temp = kernel[idxFeatureMapPrev][i * ksize + j];
          if(dOrder == 1)
            dErrorFeatureMapPrev[(row0 + i) * isize + (j + col0)] += dError[k] * temp;
          else
            dErrorFeatureMapPrev[(row0 + i) * isize + (j + col0)] += dError[k] * temp * temp;

          //get dError wrt kernel wights
          temp = valueFeatureMapPrev[(row0 + i) * isize + (j + col0)];
          if(dOrder == 1)
            dErr_wrtw[idxFeatureMapPrev][i * ksize + j] += dError[k] * temp;
          else
            dErr_wrtw[idxFeatureMapPrev][i * ksize + j] += dError[k] * temp * temp;
        }
      }
      k++;
    }
  }
}

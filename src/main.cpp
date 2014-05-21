#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <assert.h>
#include "CNN.h"

void printImage(double * img, size_t R, size_t C){
	for (size_t i = 0; i < R; i++){
		for (size_t j = 0; j < C; j++){
			printf("%c", img[i*C + j] > 0 ? '1' : '0');
		}
		printf("\n");
	}
}

void BtoLEndian32(void * mem){
	uint32_t data = *((uint32_t*)mem);
	uint32_t newData =
		((data >> 24) & 0x000000FF) |
		((data >>  8) & 0x0000FF00) |
		((data <<  8) & 0x00FF0000) |
		((data << 24) & 0xFF000000);
	*((uint32_t*)mem) = newData;
}

void loadImageSet(
	// INPUTS
	char * FName,
	// OUTPUTS
	size_t &count,
	size_t &imageR, 
	size_t &imageC, 
	size_t &imageSz, 
	double ** &data
){
	const int32_t magicExpected = 0x00000803;
	FILE * imagesFile = fopen(FName, "rb");
	if (!imagesFile){
		fprintf(stderr, "ERROR: Unable to open \"%s\" images file!\n", FName);
		exit(-1);
	}
	else{
		uint32_t aux;

		// Check magic number
		int32_t magic; fread(&magic, sizeof(magic), 1, imagesFile);
		BtoLEndian32(&magic);
		if (magic != magicExpected){
			fprintf(stderr, "ERROR: Magic number mismatch reading image file \"%s\"! Expected 0x%X, got 0x%X\n", FName, magicExpected, magic); 
			exit(-1); 
		}

		// Image count
		fread(&aux, sizeof(aux), 1, imagesFile);
		BtoLEndian32(&aux);
		count = aux;

		// Image dims (rows, cols)
		fread(&aux, sizeof(aux), 1, imagesFile);
		BtoLEndian32(&aux);
		imageR = aux;
		fread(&aux, sizeof(aux), 1, imagesFile);
		BtoLEndian32(&aux);
		imageC = aux;
		imageSz = imageR * imageC;

		// The image data
		data = (double**)malloc(count*sizeof(double*));
		unsigned char * buff = (unsigned char*)malloc(imageSz*sizeof(unsigned char));
		for (size_t i = 0; i < count; i++){
			data[i] = (double*)malloc(imageSz*sizeof(double));
			fread(buff, sizeof(unsigned char), imageSz, imagesFile);

			// Set all values to be within -1 and 1 (inputs are between 0 and 255)
			for (size_t j = 0; j < imageSz; j++) data[i][j] = (buff[j] - 128.0) / 128.0;

		}
		free(buff);
	}
	fclose(imagesFile);
}

void loadLabelSet(
	// INPUTS
	char * FName,
	size_t classCnt,
	// OUTPUTS
	size_t &count,
	double ** &outputValue, // Values fed to the CNN
	size_t * &outputIndex   // Index of values, used for verification
){
	const int32_t magicExpected = 0x00000801;
	FILE * labelsFile = fopen(FName, "rb");
	if (!labelsFile){
		fprintf(stderr, "ERROR: Unable to open \"%s\" labels file!\n", FName);
		exit(-1);
	}
	else{
		uint32_t aux;

		// Check magic number
		int32_t magic; fread(&magic, sizeof(magic), 1, labelsFile);
		BtoLEndian32(&magic);
		if (magic != magicExpected){
			fprintf(stderr, "ERROR: Magic number mismatch reading labels file \"%s\"! Expected 0x%X, got 0x%X\n", FName, magicExpected, magic);
			exit(-1);
		}

		// Image count
		fread(&aux, sizeof(aux), 1, labelsFile);
		BtoLEndian32(&aux);
		count = aux;

		// The label data
		unsigned char * buff = (unsigned char*)malloc(count*sizeof(unsigned char));
		fread(buff, sizeof(unsigned char), count, labelsFile); 
		outputValue = (double**)malloc(count*sizeof(double*));
		outputIndex = (size_t*)malloc(count*sizeof(size_t));
		for (size_t i = 0; i < count; i++){
			
			// Sanitize input
			if (buff[i] >= classCnt){ fprintf(stderr, "ERROR: Specified class (%ui) is above maximum range (%ui) at input %i!\n", buff[i], classCnt, i); exit(-1); }
			
			outputIndex[i] = buff[i];
			outputValue[i] = (double*)malloc(classCnt*sizeof(double));
			for (size_t j = 0; j < classCnt; j++) outputValue[i][j] = -1.0;
			outputValue[i][buff[i]] = 1.0;
		}
		free(buff);
	}
	fclose(labelsFile);
}

void printFormatAndExit(void){
	printf("Format:\n");
	printf("	./CNN [convolution layer count N] N*[Feature map counts] N*[Kernel sizes] N*[Step sizes] [hidden layer count M] M*[unit counts] [epochs]\n");
	exit(1);
}
void printFormatErrorAndExit(void){
	printf("ERROR: Invalid input format!");
	printFormatAndExit();
}
int main(int argc, char ** argv){

	if (argc <= 1) printFormatAndExit();
	else if (argc < 4) printFormatErrorAndExit();

	// Interpret params
	int convCnt = 0;
	int hiddenCnt = 1;
	int featureMapCnt[128];
	int kernelSizes[128];
	int stepSizes[128];
	int hiddenUnits[128];
	size_t epochs;

	{
		size_t argidx = 1;
		sscanf(argv[argidx++], "%i", &convCnt);
		for (size_t i = 0; i < convCnt; i++){
		if (argidx >= argc) printFormatErrorAndExit();
		sscanf(argv[argidx++], "%i", featureMapCnt + i);
		}
		for (size_t i = 0; i < convCnt; i++){
		if (argidx >= argc) printFormatErrorAndExit();
		sscanf(argv[argidx++], "%i", kernelSizes + i);
		}
		for (size_t i = 0; i < convCnt; i++){
		if (argidx >= argc) printFormatErrorAndExit();
		sscanf(argv[argidx++], "%i", stepSizes + i);
		}
		if (argidx >= argc) printFormatErrorAndExit();
		sscanf(argv[argidx++], "%i", &hiddenCnt);
		for (size_t i = 0; i < hiddenCnt; i++){
		if (argidx >= argc) printFormatErrorAndExit();
		sscanf(argv[argidx++], "%i", hiddenUnits + i);
		}
		if (argidx >= argc) printFormatErrorAndExit();
		sscanf(argv[argidx++], "%i", &epochs);
		if(argidx != argc) printFormatErrorAndExit();
	}
	
	// Constants
	const size_t classCnt = 10;       // Digit recognition -> 10 classes
	const double alphaInit = 0.00005; // Initial value for alpha
	const double alphaDecay = 1.0;    // At the end of each epoch, alpha gets *= by this value  
	const size_t hessianSamples = 500;

	// Load dataset
	printf("Loading dataset... ");
	double ** train;
	double ** trainOut;
	size_t * trainLabel;
	size_t trainCnt;
	double ** test;
	double ** testOut;
	size_t * testLabel;
	size_t testCnt;
	size_t imageR;
	size_t imageC;
	size_t imageSz;
	//      Load training set
	size_t aux;
	loadImageSet((char*) "data/train-images", aux, imageR, imageC, imageSz, train);
	loadLabelSet((char*) "data/train-labels", classCnt, trainCnt, trainOut, trainLabel);
	if (aux != trainCnt){ fprintf(stderr, "ERROR: Train set size missmatch! %i images but %i labels.", aux, trainCnt); exit(-1); }	
	//       Load test set
	size_t aux_imageR, aux_imageC, aux_imageSz;
	loadImageSet((char*) "data/test-images", aux, aux_imageR, aux_imageC, aux_imageSz, test);
	loadLabelSet((char*) "data/test-labels", 10, testCnt, testOut, testLabel);
	if (aux != testCnt){ fprintf(stderr, "ERROR: Test set size missmatch! %i images but %i labels.", aux, trainCnt); exit(-1); }
	if (aux_imageR != imageR || aux_imageC != imageC || aux_imageSz != imageSz){
		fprintf(stderr, "ERROR: Image dims missmatch! Dims in training are (%i * %i = %i), whereas in test they are (%i * %i = %i)",
			imageR, imageC, imageSz, aux_imageR, aux_imageC, aux_imageSz);
		exit(-1);
	}
	if (imageR != imageC){
		fprintf(stderr, "ERROR: Image is not a square (%i x %i)\n", imageR, imageC);
		exit(-1);
	}
	printf("Done!\n");

	// Trick the program into thinking it has less samples than it does
	trainCnt = 10000; 
	testCnt = 1000;    

	printf("Got %i training samples and %i test samples, with image dims %i x %i = %i\n", trainCnt, testCnt, imageR, imageC, imageSz);

	// Initialize network resources
	printf("Initializing network... ");

	//       LENET-based params
	CCNN * Net = new CCNN(convCnt, hiddenCnt, featureMapCnt, kernelSizes, stepSizes, hiddenUnits, classCnt, imageR);
	Net->LoadWeightsRandom();
	double * calcOut = (double*)malloc(classCnt*sizeof(double));
	printf("Done!\n");

	// Train Network
	printf("Training has commenced.\n");
	double alpha = alphaInit;
	for (int i = 0; i < epochs; i++){
		size_t errorCnt = 0;

		//Net->CalculateHessian();
		for (size_t j = 0; j < trainCnt; j++){
			printf("%i-%i of %i-%i\r", i, j, epochs - 1, trainCnt - 1);

			// Compute and train the network
			int predicted = Net->Calculate(train[j], calcOut);
			Net->BackPropagate(trainOut[j], alpha);

			// Compute the output and check for errors
			errorCnt += trainLabel[j] != predicted;
		}

		// Converge alpha so as to refine the search in kernel-space
		alpha *= alphaDecay;

		// Print epoch stats
		printf("Epoch %i had %i errors (%.3lf%% accuracy)\n", i, errorCnt, 100.0 - 100.0*errorCnt/trainCnt);
	}

	// Test Network's recognition
	printf("Testing has commenced.\n");
	{
		size_t errorCnt = 0;
		size_t * errorByClass = (size_t*)calloc(classCnt, sizeof(size_t));
		size_t * countByClass = (size_t*)calloc(classCnt, sizeof(size_t));
		for (size_t i = 0; i < testCnt; i++){
			int pred = Net->Calculate(test[i], calcOut);
			bool error = testLabel[i] != pred; //  maxArrayIdx(calcOut, classCnt);
			errorCnt += error ? 1 : 0;
			errorByClass[testLabel[i]] += error ? 1 : 0;
			countByClass[testLabel[i]]++;
		}
		printf("Test had %i errors (%.3lf%% accuracy)\n", errorCnt, 100.0 - 100.0*errorCnt / testCnt);
		printf("Errors, by class, were:\n");
		for (size_t i = 0; i < classCnt; i++)
			printf("%i	%i/%i	(%.3lf%% accuracy)\n", i, errorByClass[i], countByClass[i], 100.0 - 100.0*errorByClass[i] / countByClass[i]);
		free(countByClass);
		free(errorByClass);
	}

	// Free resources and exit
	delete Net;
	free(calcOut);

	for (size_t i = 0; i < trainCnt; i++){
		free(train[i]);
		free(trainOut[i]);
	}
	for (size_t i = 0; i < testCnt; i++){
		free(test[i]);
		free(testOut[i]);
	}
	
	return 0;
}
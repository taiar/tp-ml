dir=$(shell pwd)/data/
MKDIR_P = mkdir -p
RMDIR = rm -rf
CDDIR = cd
GETFILE = wget
F1 = https://dl.dropboxusercontent.com/u/55072719/test-images
F2 = https://dl.dropboxusercontent.com/u/55072719/test-labels
F3 = https://dl.dropboxusercontent.com/u/55072719/train-images
F4 = https://dl.dropboxusercontent.com/u/55072719/train-labels

entradas:
	if [ ! -d ${dir} ]; then \
	${MKDIR_P} ${dir} ;\
	${CDDIR} ${dir} ;\
	${GETFILE} ${F1} ;\
	${GETFILE} ${F2} ;\
	${GETFILE} ${F3} ;\
	${GETFILE} ${F4} ;\
	fi ;\

clean_entradas:
	${RMDIR} ${dir}

clean:
	rm -f cnn *.h.gch *.o

cleanall: clean clean_entradas

cnn: CNN.cpp CNN.h
	g++ -c CNN.cpp CNN.h -Wall -lm

main: entradas cnn
	g++ -o cnn main.cpp CNN.o -Wall -lm

test: main
	./cnn 2 13 5 5 5 2 2 1 100 10

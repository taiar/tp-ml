dir=$(shell pwd)/data/
MKDIR_P = mkdir -p
RMDIR = rm -rf
CDDIR = cd
GETFILE = wget
F1 = https://dl.dropboxusercontent.com/u/55072719/test-images
F2 = https://dl.dropboxusercontent.com/u/55072719/test-labels
F3 = https://dl.dropboxusercontent.com/u/55072719/train-images
F4 = https://dl.dropboxusercontent.com/u/55072719/train-labels
CFLAGS = -lm -O8

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
	rm -f cnn *.h.gch *.o deslocamento aprendendo

cleanall: clean clean_entradas

cnn: CNN.cpp CNN.h
	g++ -c CNN.cpp CNN.h ${CFLAGS}

main: entradas cnn
	g++ -o cnn main.cpp CNN.o ${CFLAGS}

deslocamento: entradas cnn
	g++ -o deslocamento exp_deslocamento.cpp CNN.o ${CFLAGS}

aprendendo: entradas cnn
	g++ -o aprendendo exp_aprendendo.cpp CNN.o ${CFLAGS}

taprend: aprendendo
	./aprendendo 2 13 5 5 5 2 2 1 100 10

tdesloc: deslocamento
	./deslocamento 2 13 5 5 5 2 2 1 100 10

desloc1: deslocamento
	./deslocamento 2 1 4 4 5 2 2 1 100 10

test: main
	./cnn 2 13 5 5 5 2 2 1 100 10

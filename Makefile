main:main.o classifier.o data_process.o
	g++ main.o classifier.o data_process.o -o main.out
main.o:main.cpp
	g++ -w -c main.cpp
classifier.o:classifier.cpp
	g++ -w -c classifier.cpp
data_process.o:data_process.cpp
	g++ -w -c data_process.cpp
clean:
	if [ -e main.out ]; then rm main.out; fi
	if [ -e *.o ]; then rm *.o; fi

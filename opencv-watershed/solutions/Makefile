CXX = g++
CXXFLAGS = -I/usr/include/opencv4 -DDEBUG -g # -Wall
LDFLAGS = -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

all: build/task1 # build/task2 # build/task3

build/task1: task1.o
	$(CXX) $< -o $@ $(LDFLAGS)

# build/task2: task2.o
# 	$(CXX) $< -o $@ $(LDFLAGS)

# build/task3: task3.o
#	$(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o build/task1 build/task2 build/task3

.PHONY: all clean
CXX = g++
CXXFLAGS = -I/usr/include/opencv4 -DVERBOSE # -DDEBUG -g # -Wall
LDFLAGS = -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

SRC = task1.cpp task2.cpp task3.cpp
OBJ = $(SRC:%.cpp=build/%.o)
HEADERS = watershed_utils.h # Add other common headers here if needed

all: build build/task1 build/task2 build/task3

build:
	mkdir -p build

build/task1: build/task1.o
	$(CXX) $< -o $@ $(LDFLAGS)

build/task2: build/task2.o
	$(CXX) $< -o $@ $(LDFLAGS)

build/task3: build/task3.o
	$(CXX) $< -o $@ $(LDFLAGS)

# Put object files in build/
build/%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f build/*.o build/task1 build/task2 build/task3

.PHONY: all clean build
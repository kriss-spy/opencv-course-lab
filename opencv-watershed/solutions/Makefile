CXX = g++
CXXFLAGS = -I/usr/include/opencv4 -DDEBUG
LDFLAGS = -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
SRC = task1.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = build/task1

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
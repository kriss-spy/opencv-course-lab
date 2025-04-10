CXX = g++
CXXFLAGS = -I/usr/include/opencv4
LDFLAGS = -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
SRC = cv2-watershed.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = build/cv2-watershed

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
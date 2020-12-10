UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
# Linux
	CC = g++
	SHARED_FLAG = -shared
else
	ifeq ($(UNAME), Darwin)
# OS X
		CC = g++
		SHARED_FLAG = -dynamiclib
	else
		CC = g++
	endif
endif

CXXFLAGS = -Wall -fopenmp -O3 -march=native -DNDEBUG -std=c++11

INCLUDES = -I${EIGEN_DIR} -I${BOOST_DIR} -I/usr/include/eigen3
LDFLAGS = -lpthread

all: Grid.o Grid_ctypes.o FDTD.o FDTD_2D.o FDTD_2D_CYL.o Grid.cpp Grid.hpp Grid_ctypes.cpp Grid_ctypes.hpp fdtd.cpp fdtd.hpp fdtd_2d.cpp fdtd_2d.hpp fdtd_2d_cyl_v2.cpp fdtd_2d_cyl_v2.hpp
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o Grid.so Grid.o Grid_ctypes.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o FDTD.so FDTD.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o FDTD_2D.so FDTD_2D.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o FDTD_2D_CYL.so FDTD_2D_CYL.o ${LDFLAGS}
	cp Grid.so ../emopt2
	cp FDTD.so ../emopt2
	cp FDTD_2D.so ../emopt2
	cp FDTD_2D_CYL.so ../emopt2

Grid.o: Grid.cpp Grid.hpp
	${CC} -c -fPIC Grid.cpp ${CXXFLAGS} -o Grid.o ${INCLUDES}

Grid_ctypes.o: Grid.cpp Grid.hpp Grid_ctypes.cpp Grid_ctypes.hpp
	${CC} -c -fPIC Grid_ctypes.cpp ${CXXFLAGS} -o Grid_ctypes.o ${INCLUDES}

FDTD.o: fdtd.cpp fdtd.hpp
	${CC} -c -fPIC fdtd.cpp ${CXXFLAGS} -o FDTD.o

FDTD_2D.o: fdtd_2d.cpp fdtd_2d.hpp
	${CC} -c -fPIC fdtd_2d.cpp ${CXXFLAGS} -o FDTD_2D.o

FDTD_2D_CYL.o: fdtd_2d_cyl_v2.cpp fdtd_2d_cyl_v2.hpp
	${CC} -c -fPIC fdtd_2d_cyl_v2.cpp ${CXXFLAGS} -o FDTD_2D_CYL.o

clean:
	rm -f ./*.o
	rm -f ./*.so
	rm -f ../emopt2/*.so
	rm -f ../emopt2/*.pyc


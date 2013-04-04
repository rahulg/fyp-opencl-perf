HAS_CLANG=$(shell which clang)
ifeq ($(HAS_CLANG),)
	CC=gcc
else
	CC=clang
endif

HAS_CLANGXX=$(shell which clang++)
ifeq ($(HAS_CLANGXX),)
	CXX=g++
else
	CXX=clang++
endif

CFLAGS?=
CXXFLAGS?=-std=c++11
LIBS=-L../../derpcl -lderpcl
INCL=-I../../

SYSTEM=$(shell uname -s)
ifeq ($(SYSTEM), Linux)
LIBS += -lOpenCL
else
LIBS += -framework OpenCL
CXXFLAGS += -stdlib=libc++
endif

WIDTH?=1920
HEIGHT?=1080

.PHONY: all debug force_look clean

all: CXXFLAGS += -O4
all: OPT += -fvectorize -fslp-vectorize -DWIDTH=$(WIDTH) -DHEIGHT=$(HEIGHT)
all:
	$(CXX) $(CXXFLAGS) $(INCL) $(OPT) opencl.cc $(LIBS) -o opencl
	$(CXX) $(CXXFLAGS) $(INCL) $(OPT) cpp.cc $(LIBS) -o cpp

debug: CXXFLAGS += -O0 -DDEBUG -g
debug: OPT += -fvectorize -fslp-vectorize -DWIDTH=$(WIDTH) -DHEIGHT=$(HEIGHT)
debug:
	$(CXX) $(CXXFLAGS) $(INCL) $(OPT) opencl.cc $(LIBS) -o opencl
	$(CXX) $(CXXFLAGS) $(INCL) $(OPT) cpp.cc $(LIBS) -o cpp

clean: force_look
	-rm -f opencl cpp
	-rm -rf *.dSYM

force_look:
	@true

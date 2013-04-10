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

.PHONY: all debug force_look clean

all: CXXFLAGS += -O4
all: $(BUILDS)

debug: CXXFLAGS += -O0 -DDEBUG -g
debug: $(BUILDS)

%: CXXFLAGS += -O4
%: OPT += -fvectorize -fslp-vectorize -DFWIDTH=$@
%:
	$(CXX) $(CXXFLAGS) $(INCL) $(OPT) vec.cc $(LIBS) -o vec

clean: force_look
	-rm -f opencl cpp
	-rm -rf *.dSYM

force_look:
	@true

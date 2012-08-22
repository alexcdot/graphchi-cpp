INCFLAGS = -I/usr/local/include/ -I./src/

CPP = g++
CPPFLAGS = -g -O3 $(INCFLAGS)  -fopenmp -Wall -Wno-strict-aliasing
DEBUGFLAGS = -g -ggdb $(INCFLAGS)
HEADERS=$(wildcard *.h**)


all: apps tests 
apps: example_apps/connectedcomponents example_apps/pagerank example_apps/pagerank_functional example_apps/communitydetection example_apps/trianglecounting
als: example_apps/matrix_factorization/als_edgefactors  example_apps/matrix_factorization/als_vertices_inmem
tests: tests/basic_smoketest tests/bulksync_functional_test


clean:
	@rm -rf bin/*

sharder_basic: src/preprocessing/sharder_basic.cpp $(HEADERS)
	$(CPP) $(CPPFLAGS) src/preprocessing/sharder_basic.cpp -o bin/sharder_basic

example_apps/% : example_apps/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Iexample_apps/ $@.cpp -o bin/$@


myapps/% : myapps/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Imyapps/ $@.cpp -o bin/$@

tests/%: src/tests/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) src/$@.cpp -o bin/$@	


graphlab_als: example_apps/matrix_factorization/graphlab_gas/als_graphlab.cpp
	$(CPP) $(CPPFLAGS) example_apps/matrix_factorization/graphlab_gas/als_graphlab.cpp -o bin/graphlab_als

cf: toolkits/collaborative_filtering/*
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Itoolkits/collaborative_filtering/ toolkits/collaborative_filtering/wals.cpp  -o bin/wals
	$(CPP) $(CPPFLAGS) -Itoolkits/collaborative_filtering/ toolkits/collaborative_filtering/svdpp.cpp  -o bin/svdpp
	$(CPP) $(CPPFLAGS) -Itoolkits/collaborative_filtering/ toolkits/collaborative_filtering/als.cpp  -o bin/als
	$(CPP) $(CPPFLAGS) -Itoolkits/collaborative_filtering/ toolkits/collaborative_filtering/sgd.cpp  -o bin/sgd
	$(CPP) $(CPPFLAGS) -Itoolkits/collaborative_filtering/ toolkits/collaborative_filtering/biassgd.cpp  -o bin/biassgd

docs: */**
	doxygen conf/doxygen/doxygen.config


	

	
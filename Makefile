CC := c++    # use mpiCC when not on okeanos
MPICC := mpicc    # use mpiCC when not on okeanos
LFLAGS := -std=c++17 -g
ALL := bin/dijkstra bin/sssp bin/handle_test

all : $(ALL)

bin/dijkstra: dijkstra/dijkstra.cpp
	$(CC) $(LFLAGS) -o $@ $< 

bin/handle_test: handle_test/handle_test.cpp
	$(CC) $(LFLAGS) -o $@ $< 

bin/sssp: sssp/sssp.cpp
	$(MPICC) $(LFLAGS) -o $@ $< 

clean :
	rm -f $(ALL)

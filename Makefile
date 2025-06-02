CC := c++    # use mpiCC when not on okeanos
MPICC := mpicc    # use mpiCC when not on okeanos
LFLAGS := -std=c++17 -g
ALL := bin/dijkstra

all : $(ALL)

bin/dijkstra: dijkstra/dijkstra.cpp
	$(CC) $(LFLAGS) -o $@ $< 

bin/sssp: sssp/sssp.cpp
	$(MPICC) $(LFLAGS) -o $@ $< 

clean :
	rm -f $(ALL)

CC := CC    # use mpiCC when not on okeanos
MPICC := CC    # use mpiCC when not on okeanos
LFLAGS := -std=c++17 -g
ALL := sssp

all : $(ALL)

sssp: ../../sssp/sssp.cpp
	$(MPICC) $(LFLAGS) -o $@ $< 

clean :
	rm -f $(ALL)
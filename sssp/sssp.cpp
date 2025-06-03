#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include "mpi.h"

using std::cerr;
using std::string;
using std::vector;
using std::unordered_map;
using std::pair;
using std::max;
using std::min;

constexpr long long inf = 5'000'000'000'000'000'000;
static_assert(inf > 0, "Bad inf size");
constexpr long long delta = 10;
static_assert(delta > 0, "Bad delta size");

int n;
int start, end;

int numProcesses;
int myRank;

MPI_Comm intra_node_comm;
int intraRank, intraNum;

int max_length;
vector<int> ends; ///< The last vertices kept in each process

// mapping from known outside vertices to their (process, local_id)
unordered_map<int, pair<int, int>> outside_address;

class Local {
private:
	int index;	

	template<class T>
	friend class LocalVector;

public:
	Local(): index(0) {}
	Local(const Local& other): index(other.index) {}
	Local(int ind): index(ind) {}

	int global() const {
		return start + index;
	}
};

bool is_local(const int index) {
	return start <= index && index <= end;
} 

template<class T>
class LocalVector {
private:
	vector<T> data;
public:
	LocalVector(): data() {}
	LocalVector(int n, const T def = {}): data(n, def) {}

	vector<T>* operator->() {
		return &data;
	}

	const vector<T>* operator->() const {
		return &data;
	}

	T& operator[](const int index) {
		assert(is_local(index));
		return data[Local(index).index];
	}

	const T& operator[](const int index) const {
		assert(is_local(index));
		return data[Local(index).index];
	}

	T& operator[](const Local index) {
		return data[index.index];
	}

	const T& operator[](const Local index) const {
		return data[index.index];
	}
};

LocalVector<vector<pair<int, long long>>> edges(n);

void setup() {
	ends.resize(numProcesses);
	// Collect common start/end of vertices of each process
	if (myRank == 0) {
		ends[0] = end;
		for(int i = 1; i < numProcesses; i++) {
			MPI_Recv(
				&ends[i], 
				1,
				MPI_INT, 
				i,
				0,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE
			);
		}
		max_length = end - start + 1;
		for(int i = 1; i < numProcesses; i++) {
			max_length = max(max_length, ends[i] - ends[i - 1]);
		}
	} else {
		MPI_Send(
			&end, 
			1, 
			MPI_INT, 
			0, 
			0, 
			MPI_COMM_WORLD
		);
	}
	// Get common values
	MPI_Bcast(
		&max_length, 
		1, 
		MPI_INT, 
		0,
		MPI_COMM_WORLD
	);
	MPI_Bcast(
		ends.data(), 
		numProcesses, 
		MPI_INT, 
		0,
		MPI_COMM_WORLD
	);
	for(int src = start; src <= end; src++) {
		for(auto [dest, len]: edges[src]) {
			if (!is_local(dest)) {
				int ind = std::lower_bound(ends.begin(), ends.end(), dest) - ends.begin();
				if (ind == 0) {
					outside_address[ind] = {0, dest};
				} else {
					outside_address[ind] = {ind, dest - ends[ind - 1]};
				}
			}
		}
	}
	// Define the intra node split communicator
	MPI_Comm_split_type(
		MPI_COMM_WORLD, 
		MPI_COMM_TYPE_SHARED, 
		0, 
		MPI_INFO_NULL, 
		&intra_node_comm
	);
    MPI_Comm_size(intra_node_comm, &intraNum);
    MPI_Comm_rank(intra_node_comm, &intraRank);
	/**
	 * Maybe add inter and intra-node load balancing
	 */
	
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		cerr << "Usage: " << string(argv[0]) << " input_file output_file"; 
		return 1;
	}

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	std::fstream in(string(argv[1]), std::ios_base::in);

	in >> n >> start >> end;
	edges->resize(n);
	int x, y;
	long long len;
	while(in >> x) {
		in >> y >> len;
		if (is_local(x)) {
			edges[x].emplace_back(y, len);
		}
		if (is_local(y)) {
			edges[y].emplace_back(x, len);
		}
	}
	in.close();

	setup();

	std::fstream out(string(argv[2]), std::ios_base::out);
	out << "Global:" << myRank << "/" << numProcesses << "\n";
	out << "Local:" << intraRank << "/" << intraNum << "\n";
	out << "max_length: " << max_length << "\n";
	for(int i = 0; i < numProcesses; i++) {
		out << ends[i] << " ";
	}
	out << "\n";
	for(auto [src, rest]: outside_address) {
		out << src << " -> " << rest.first << ", " << rest.second << "\n";
	}
	out.close();
}
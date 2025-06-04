#include <iostream>
#include <sstream>
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
constexpr long long delta = inf;
static_assert(delta > 0, "Bad delta size");

std::fstream err;

int n;
int start, end;
int length;

int phases = 0, non_phase_steps = 0, local_relaxations = 0, non_phase_comms = 0;

int numProcesses;
int myRank;

MPI_Comm intra_node_comm;
int intraRank, intraNum;

int max_length;
vector<int> ends; ///< The last vertices kept in each process

// mapping from known outside vertices to their (process, local_id)
unordered_map<int, pair<int, int>> outside_address;

bool is_local(const int index) {
	return start <= index && index <= end;
} 

struct from_local_t {};
from_local_t from_local;

class Local {
private:
	int index;	

	template<class T>
	friend class LocalVector;

public:
	Local(): index(0) {}
	Local(const Local& other): index(other.index) {}
	Local(int ind): index(ind - start) {}
	Local(from_local_t, int ind): index(ind) {
		assert(is_local(ind));
	}

	int global() const {
		return start + index;
	}
};

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

	vector<T>& operator*() {
		return data;
	}

	const vector<T>& operator*() const {
		return data;
	}


	typename vector<T>::reference operator[](const int index) {
		assert(is_local(index));
		return data[Local(index).index];
	}

	typename vector<T>::const_reference operator[](const int index) const {
		assert(is_local(index));
		return data[Local(index).index];
	}

	typename vector<T>::reference operator[](const Local index) {
		return data[index.index];
	}

	typename vector<T>::const_reference operator[](const Local index) const {
		return data[index.index];
	}
};

LocalVector<vector<pair<int, long long>>> edges(n);
LocalVector<long long> dist; ///< Local distances from 0
LocalVector<bool> settled;
LocalVector<bool> changed;
LocalVector<long long> vertex_window_data;

MPI_Win vertex_window;

void deltaEpochSetup(long long base) {
	for(int src = start; src <= end; src++) {
		if (dist[src] >= base && dist[src] < base + delta) {
			changed[src] = true;
		} else {
			changed[src] = false;
		}
	}
}

bool deltaSingleStep(long long base) {
	// err << "Starting delta single step with base: " << base << std::endl;
	bool was_changed = false, global_changed = false;
	LocalVector<bool> active(length, false);
	for(int src = start; src <= end; src++) {
		active[src] = changed[src];
		changed[src] = false;
	}

	for(auto& v: *vertex_window_data) {
		v = inf;
	}

	MPI_Win_fence(MPI_MODE_NOPRECEDE, vertex_window);

	for(int src = start; src <= end; src++) {
		if (active[src]) {
			for(auto [dest, len]: edges[src]) {
				long long new_dist = dist[src] + len;
				if (is_local(dest)) {
					local_relaxations++;
					if (new_dist < dist[dest]) {
						dist[dest] = new_dist;
						if (new_dist < base + delta) {
							changed[dest] = true;
							was_changed = true;
						}
					}
				} else {
					auto [destRank, destId] = outside_address[dest];
					non_phase_comms++;
					MPI_Accumulate(
						&new_dist, 
						1, 
						MPI_LONG_LONG, 
						destRank, 
						destId, 
						1, 
						MPI_LONG_LONG, 
						MPI_MIN, 
						vertex_window
					);
				}
			}
		}
	}

	MPI_Win_fence(MPI_MODE_NOSUCCEED, vertex_window);

	for(int src = start; src <= end; src++) {
		if (vertex_window_data[src] < dist[src]) {
			dist[src] = vertex_window_data[src];
			if (dist[src] < base + delta) {
				changed[src] = true; 
				was_changed = true;
			}
		}	
	}

	MPI_Allreduce(
		&was_changed, 
		&global_changed, 
		1, 
		MPI_CXX_BOOL, 
		MPI_LOR, 
		MPI_COMM_WORLD
	);

	return global_changed;
}

bool deltaEpoch() {
	long long min_dist = inf, global_min_dist = inf;
	for(int src = start; src <= end; src++) {
		if (!settled[src]) min_dist = min(min_dist, dist[src]);
	}
	MPI_Allreduce(
		&min_dist, 
		&global_min_dist, 
		1, 
		MPI_LONG_LONG, 
		MPI_MIN, 
		MPI_COMM_WORLD
	);
	if (global_min_dist == inf) return false;

	deltaEpochSetup(global_min_dist);

	non_phase_steps++;
	while(deltaSingleStep(global_min_dist)) {
		non_phase_steps++;
	}

	for(int src = start; src <= end; src++) {
		if (dist[src] < global_min_dist + delta) {
			settled[src] = true;
		}
	}

	return true;
}

bool bellmanFordStep() {
	// err << "Starting bellman-ford in phase: " << phases << std::endl;
	LocalVector<bool> new_changed(length, false);
	bool was_changed = false;
	bool global_changed = false;

	for(auto& v: *vertex_window_data) {
		v = inf;
	}

	MPI_Win_fence(MPI_MODE_NOPRECEDE, vertex_window);

	for(int src = start; src <= end; src++) {
		if (changed[src]) {
			for(auto [dest, len]: edges[src]) {
				long long new_dist = dist[src] + len;
				if (is_local(dest)) {
					local_relaxations++;
					if (new_dist < dist[dest]) {
						dist[dest] = new_dist;
						new_changed[dest] = true;
						was_changed = true;
					}
				} else {
					auto [destRank, destId] = outside_address[dest];
					non_phase_comms++;
					MPI_Accumulate(
						&new_dist, 
						1, 
						MPI_LONG_LONG, 
						destRank, 
						destId, 
						1, 
						MPI_LONG_LONG, 
						MPI_MIN, 
						vertex_window
					);
				}
			}
		}
	}

	MPI_Win_fence(MPI_MODE_NOSUCCEED, vertex_window);

	for(int src = start; src <= end; src++) {
		if (vertex_window_data[src] < dist[src]) {
			dist[src] = vertex_window_data[src];
			new_changed[src] = true; 
			was_changed = true;
		}	
	}

	MPI_Allreduce(
		&was_changed, 
		&global_changed, 
		1, 
		MPI_CXX_BOOL, 
		MPI_LOR, 
		MPI_COMM_WORLD
	);
	for(auto src = start; src <= end; src++) {
		changed[src] = new_changed[src];
	}

	return global_changed;
}

void setup() {
	length = end - start + 1;
	dist->resize(length);
	settled->resize(length);
	changed->resize(length);
	ends.resize(numProcesses);
	vertex_window_data->resize(length);
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
	// Calculate outside mapping
	for(int src = start; src <= end; src++) {
		for(auto [dest, len]: edges[src]) {
			if (!is_local(dest)) {
				int ind = std::lower_bound(ends.begin(), ends.end(), dest) - ends.begin();
				if (ind == 0) {
					outside_address[dest] = {0, dest};
				} else {
					outside_address[dest] = {ind, dest - ends[ind - 1] - 1};
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
	// Setup window
	MPI_Win_create(
		vertex_window_data->data(), 
		sizeof(long long) * length, 
		sizeof(long long), 
		MPI_INFO_NULL, 
		MPI_COMM_WORLD, 
		&vertex_window
	);
	for(int src = start; src <= end; src++) {
		changed[src] = false;
		dist[src] = inf;
	}
	if (myRank == 0) {
		changed[0] = true;
		dist[0] = 0;
	}
	/**
	 * Maybe add inter and intra-node load balancing
	 */
}

void read(string file) {
	std::fstream in(file, std::ios_base::in);

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
}

void write_out(string file) {
	std::fstream out(file, std::ios_base::out);

	for(auto len: *dist) {
		out << len << "\n";		
	}
	
	out.close();
}

void write_debug(string file) {
	std::fstream out(string(file), std::ios_base::out);
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

void write_info() {
	err << "Number of phases: " << phases << std::endl;
	err << "Number of non phase steps: " << non_phase_steps << std::endl;
	err << "Non phase related communications: " << non_phase_comms << std::endl;
	err << "Local relaxations: " << local_relaxations << std::endl;
}

void finish() {
	MPI_Win_free(&vertex_window);
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		cerr << "Usage: " << string(argv[0]) << " input_file output_file"; 
		return 1;
	}

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	std::stringstream s;
	s << "error/" << myRank << ".err";
	err.open(s.str(), std::ios_base::out);

	read(argv[1]);
	setup();

	phases++;
	while(deltaEpoch()) {
		phases++;
	}

	write_info();
	write_out(argv[2]);

	err.close();

	finish();
}
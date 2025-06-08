#include <iostream>
#include <sstream>
#include <set>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "mpi.h"

using std::cerr;
using std::string;
using std::vector;
using std::unordered_map;
using std::pair;
using std::tuple;
using std::set;
using std::unordered_set;
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

template<class T, template<typename> typename S>
vector<vector<T>> shareWithAll(const vector<S<T>>& out_requests, const MPI_Datatype& mpi_type) {
	// setup out request information
	vector<T> flat_out_requests;
	vector<int> out_request_addresses(numProcesses);

	vector<int> out_amounts(numProcesses, 0);
	vector<int> in_amounts(numProcesses, 0);
	int out_it = 0;
	for(int rank = 0; rank <= numProcesses; rank++) {
		out_request_addresses[rank] = out_it;
		for(auto req: out_requests[rank]) {
			flat_out_requests.push_back(req);
			out_it++;
		}
		out_amounts[rank] = out_it - out_request_addresses[rank];
	}

	// communicate request sizes
	MPI_Alltoall(
		out_amounts.data(), 
		1, MPI_INT, 
		in_amounts.data(), 
		1, MPI_INT, 
		MPI_COMM_WORLD);

	// setup in request information
	vector<T> flat_in_requests;
	vector<int> in_request_addresses(numProcesses);
	int sum = 0;
	for(int rank = 0; rank < numProcesses; rank++) {
		in_request_addresses[rank] = sum;
		sum += in_amounts[rank];
	}
	flat_in_requests.resize(sum);

	MPI_Alltoallv(
		flat_out_requests.data(), 
		out_amounts.data(), 
		out_request_addresses.data(), 
		mpi_type, 
		flat_in_requests.data(), 
		in_amounts.data(), 
		in_request_addresses.data(), 
		mpi_type, 
		MPI_COMM_WORLD
	);

	vector<vector<T>> in_requests(numProcesses);
	for(int rank = 0; rank < numProcesses; rank++) {
		in_requests.resize(in_amounts[rank]);
		for(int i = 0; i < in_amounts[rank]; i++) {
			in_requests[rank][i] = flat_in_requests[in_request_addresses[rank] + i];
		}
	}
	return in_requests;
}

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

struct Target{
	long long length;
	int dest;
	int destRank;
	int destId;

	bool operator<(const Target& other) {
		if (length != other.length) return length < other.length;
		return dest < other.dest;
	}

	Target(int dest, long long length): length(length), dest(dest) {}

	static Target lower(long long len) {
		return Target(-1, len);
	}

	static Target upper(long long len) {
		return Target(end + 1, len);
	}
};

MPI_Comm intra_node_comm;
int intraRank, intraNum;

int max_length;
vector<int> ends; ///< The last vertices kept in each process

// mapping from known outside vertices to their (process, local_id)
unordered_map<int, pair<int, int>> outside_address;
set<pair<long long, int>> vertices_by_distance;
unordered_set<int> active;

LocalVector<vector<Target>> edges(n);
LocalVector<vector<Target>> short_edges(n);
LocalVector<vector<Target>> long_edges(n);
LocalVector<long long> dist; ///< Local distances from 0
LocalVector<bool> settled;
unordered_set<int> unsettled;
unordered_set<int> current_bucket;
// LocalVector<bool> changed;
LocalVector<long long> vertex_window_data;

inline void update_distance(int src, long long new_dist) {
	vertices_by_distance.erase({dist[src], src});
	dist[src] = new_dist;
	vertices_by_distance.insert({new_dist, src});
}

vector<int> node_window_comm; 

MPI_Win vertex_window;
MPI_Win node_window;

void deltaEpochSetup(long long base) {
	current_bucket = {};
	// Pick active vertices
	while(vertices_by_distance.size() && vertices_by_distance.begin()->first < base + delta) {
		auto x = *vertices_by_distance.begin();
		active.insert(x.second); 
		current_bucket.insert(x.second);
		vertices_by_distance.erase(vertices_by_distance.begin());
	}
	// for(auto src: vertices_by_distance) {
		// if (dist[src] >= base && dist[src] < base + delta) {
			// changed[src] = true;
		// } else {
			// changed[src] = false;
		// }
	// }
}

template<bool classification, bool pull>
bool deltaSingleStep(long long base) {
	// err << "Starting delta single step with base: " << base << std::endl;
	bool was_changed = false, global_changed = false;
	unordered_set<int> new_active;

	vector<unordered_map<int, long long>> best_update;

	for(auto src: active) {
		auto edges_begin = edges[src].begin();
		auto edges_end = edges[src].end();
		if constexpr(classification) {
			edges_begin = short_edges[src].begin();
			edges_end = std::lower_bound(short_edges[src].begin(), short_edges[src].end(), Target::lower(base + delta - dist[src]));
		}
		for(auto it = edges_begin; it != edges_end; it++) {
			auto& target = *it;
			long long new_dist = dist[src] + target.length;
			if (is_local(target.dest)) {
				local_relaxations++;
				if (new_dist < dist[target.dest]) {
					// Possibly not useful if anymore
					if (new_dist < base + delta) {
						if (dist[target.dest] >= base + delta) {
							current_bucket.insert(target.dest);
						}
						new_active.insert(target.dest);
						was_changed = true;
					}
					update_distance(target.dest, new_dist);
				}
			} else {
				// auto [destRank, destId] = outside_address[target.dest];
				non_phase_comms++;
				auto it  = best_update[target.destRank].find(target.dest);
				auto best = new_dist;
				if (it != best_update[target.destRank].end()) {
					best = min(it->second, new_dist);
				}	
				best_update[target.destRank][target.dest] = best;
			}
		}
	}

	vector<vector<int>> out_vertex(numProcesses);
	vector<vector<long long>> out_dist(numProcesses);
	
	for(int rank = 0; rank < numProcesses; rank++) {
		for(auto [dest, new_dist]: best_update[rank]) {
			out_vertex[rank].push_back(dest);
			out_dist[rank].push_back(new_dist);
		}
	}

	auto in_vertex = shareWithAll(out_vertex, MPI_INT);
	auto in_dist = shareWithAll(out_dist, MPI_LONG_LONG);

	for(int rank = 0; rank < numProcesses; rank++) {
		for(int i = 0; i < in_vertex.size(); i++) {
			auto dest = in_vertex[rank][i];
			auto new_dist = in_dist[rank][i];
			if (new_dist < dist[dest]) {
				// Possibly not useful if anymore
				if (new_dist < base + delta) {
					if (dist[dest] >= base + delta) {
						current_bucket.insert(dest);
					}
					new_active.insert(dest); 
					was_changed = true;
				}
				update_distance(dest, vertex_window_data[dest]);
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

	active = new_active;

	return global_changed;
}

template<bool classification, bool pull>
void deltaLongPhase(int base) {
	for(auto& v: *vertex_window_data) {
		v = inf;
	}

	active = {};

	for(auto src: current_bucket) {
		unsettled.erase(src);
	}

	if constexpr (classification && !pull) {
		vector<unordered_map<int, long long>> best_update;

		// Handle edges going out of the current bracket forward.
		for(auto src: current_bucket) {
			auto edges_begin = std::lower_bound(edges[src].begin(), edges[src].end(), Target::lower(base + delta - dist[src]));
			auto edges_end = edges[src].end();
			for(auto it = edges_begin; it != edges_end; it++) {
				auto& target = *it;
				auto new_dist = target.length + dist[src];

				auto mit = best_update[target.destRank].find(target.dest);
				auto best = new_dist;
				if (mit != best_update[target.destRank].end()) {
					best = min(mit->second, new_dist);
				}	
				best_update[target.destRank][target.dest] = best;
			}
		}

		vector<vector<int>> out_vertex(numProcesses);
		vector<vector<long long>> out_dist(numProcesses);
	
		for(int rank = 0; rank < numProcesses; rank++) {
			for(auto [dest, new_dist]: best_update[rank]) {
				out_vertex[rank].push_back(dest);
				out_dist[rank].push_back(new_dist);
			}
		}

		auto in_vertex = shareWithAll(out_vertex, MPI_INT);
		auto in_dist = shareWithAll(out_dist, MPI_LONG_LONG);

		for(int rank = 0; rank < numProcesses; rank++) {
			for(int i = 0; i < in_vertex.size(); i++) {
				auto dest = in_vertex[rank][i];
				auto new_dist = in_dist[rank][i];
				if (new_dist < dist[dest]) {
					active.insert(dest); 
					update_distance(dest, vertex_window_data[dest]);
				}	
			} 
		}
	} else if constexpr (pull) {
		// Gather requests
		vector<vector<tuple<int, long long, int>>> attributed_requests(numProcesses);
		vector<set<int>> out_requests(numProcesses);
		for(auto src: unsettled) {
			auto edges_begin = edges[src].begin();
			auto edges_end = std::lower_bound(edges[src].begin(), edges[src].end(), Target::lower(dist[src] - base));

			for(auto it = edges_begin; it != edges_end; it++) {
				auto& target = *it;
				if (is_local(target.dest)) {
					local_relaxations++;
					// probably useless if
					if (dist[target.dest] + target.length < dist[src]) {
						update_distance(src, dist[target.dest] + target.length);
						active.insert(src);
					}
				}
				// Probably useless if
				else if (target.length + base < dist[src]) {
					auto [destRank, destId] = outside_address[target.dest];
					out_requests[destRank].insert(target.dest);
					attributed_requests[destRank].emplace_back(src, target);
				}
			}
		}
		auto in_requests = shareWithAll(out_requests, MPI_INT);

		vector<vector<int>> out_vertices(numProcesses);
		vector<vector<long long>> out_lengths(numProcesses);
		for(auto rank = 0; rank < numProcesses; rank++) {
			for(auto req: in_requests[rank]) {
				if (current_bucket.find(req) != current_bucket.end()) {
					out_vertices[rank].push_back(req);
					out_lengths[rank].push_back(dist[req]);
				}
			}
		}
		auto in_vertices = shareWithAll(out_vertices, MPI_INT);
		auto in_lengths = shareWithAll(out_lengths, MPI_LONG_LONG);

		for(auto rank = 0; rank < numProcesses; rank++) {
			unordered_map<int, long long> partial_outside;
			for(int i = 0; i < in_vertices[rank].size(); i++) {
				partial_outside[in_vertices[rank][i]] = in_lengths[rank][i];
			}
			for(auto [src, len, dest]: attributed_requests[rank]) {
				auto it = partial_outside.find(dest);
				if (it != partial_outside.end()) {
					auto new_dist = len + it->second;
					if (new_dist < dist[src]) {
						dist[src] = new_dist;
						update_distance(src, new_dist);
						active.insert(src);
					}
				}
			}
		}
	}
}

template <bool classification, bool pull>
void deltaEpochEpilogue(int base) {

	if constexpr (classification) {
		deltaLongPhase<classification, pull>(base);
	}

	for(int src = start; src <= end; src++) {
		if (dist[src] < base + delta) {
			settled[src] = true;
		}
	}
}

template <bool classification, bool pull>
bool deltaEpoch() {
	static_assert(!pull || classification, "pull model only available if classification is enables");
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
	while(deltaSingleStep<classification, pull>(global_min_dist)) {
		non_phase_steps++;
	}

	deltaEpochEpilogue<classification, pull>(global_min_dist);

	return true;
}

// bool bellmanFordStep() {
	// // err << "Starting bellman-ford in phase: " << phases << std::endl;
	// LocalVector<bool> new_changed(length, false);
	// bool was_changed = false;
	// bool global_changed = false;

	// for(auto& v: *vertex_window_data) {
		// v = inf;
	// }

	// MPI_Win_fence(MPI_MODE_NOPRECEDE, vertex_window);

	// for(int src = start; src <= end; src++) {
		// if (changed[src]) {
			// for(auto [dest, len]: edges[src]) {
				// long long new_dist = dist[src] + len;
				// if (is_local(dest)) {
					// local_relaxations++;
					// if (new_dist < dist[dest]) {
						// dist[dest] = new_dist;
						// new_changed[dest] = true;
						// was_changed = true;
					// }
				// } else {
					// auto [destRank, destId] = outside_address[dest];
					// non_phase_comms++;
					// MPI_Accumulate(
						// &new_dist, 
						// 1, 
						// MPI_LONG_LONG, 
						// destRank, 
						// destId, 
						// 1, 
						// MPI_LONG_LONG, 
						// MPI_MIN, 
						// vertex_window
					// );
				// }
			// }
		// }
	// }

	// MPI_Win_fence(MPI_MODE_NOSUCCEED, vertex_window);

	// for(int src = start; src <= end; src++) {
		// if (vertex_window_data[src] < dist[src]) {
			// dist[src] = vertex_window_data[src];
			// new_changed[src] = true; 
			// was_changed = true;
		// }	
	// }

	// MPI_Allreduce(
		// &was_changed, 
		// &global_changed, 
		// 1, 
		// MPI_CXX_BOOL, 
		// MPI_LOR, 
		// MPI_COMM_WORLD
	// );
	// for(auto src = start; src <= end; src++) {
		// changed[src] = new_changed[src];
	// }

	// return global_changed;
// }

void setup() {
	length = end - start + 1;
	dist->resize(length);
	settled->resize(length);
	short_edges->resize(length);
	long_edges->resize(length);
	active = {0};
	ends.resize(numProcesses);
	vertex_window_data->resize(length);
	node_window_comm.resize(numProcesses);
	// Collect common start/end of vertices of each process
	if (myRank == 0) {
		ends[0] = end;
		for(int i = 1; i < numProcesses; i++) {
			MPI_Recv(
				&ends[i], 
				1, MPI_INT, 
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
			1, MPI_INT, 
			0, 
			0, 
			MPI_COMM_WORLD
		);
	}
	// Get common values
	MPI_Bcast(
		&max_length, 
		1, MPI_INT, 
		0,
		MPI_COMM_WORLD
	);
	MPI_Bcast(
		ends.data(), 
		numProcesses, MPI_INT, 
		0,
		MPI_COMM_WORLD
	);
	// Calculate outside mapping
	for(int src = start; src <= end; src++) {
		for(auto& target: edges[src]) {
			int ind = std::lower_bound(ends.begin(), ends.end(), target.dest) - ends.begin();
			if (ind == 0) {
				target.destRank = 0;
				target.destId = target.dest;
			} else {
				target.destRank = ind;
				target.destId = target.dest - ends[ind - 1] - 1;
			}
			if (target.length < delta) {
				short_edges[src].push_back(target);
			} else {
				long_edges[src].push_back(target);
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
	MPI_Win_create(
		&node_window_comm, 
		sizeof(int) * length, 
		sizeof(int), 
		MPI_INFO_NULL, 
		MPI_COMM_WORLD, 
		&node_window
	);
	for(int src = start; src <= end; src++) {
		dist[src] = inf;
	}
	if (myRank == 0) {
		dist[0] = 0;
	}
	for(int src = start; src <= end; src++) {
		vertices_by_distance.insert({dist[src], src});
	}
}

void read(string file) {
	std::fstream in(file, std::ios_base::in);

	in >> n >> start >> end;
	edges->resize(n);
	short_edges->resize(n);
	int x, y;
	long long len;
	while(in >> x) {
		in >> y >> len;
		if (is_local(x)) {
			edges[x].emplace_back(y, len);
			if (len < delta) {
				short_edges[x].emplace_back(y, len);
			} else {
				long_edges[x].emplace_back(y, len);
			}
		}
		if (is_local(y)) {
			edges[y].emplace_back(x, len);
			if (len < delta) {
				short_edges[y].emplace_back(x, len);
			} else {
				long_edges[y].emplace_back(x, len);
			}
		}
	}
	for(int src = start; src <= end; src++) {
		sort(edges[src].begin(), edges[src].end());
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
	while(deltaEpoch<true, true>()) {
		phases++;
	}

	write_info();
	write_out(argv[2]);

	err.close();

	finish();
}
#include <iostream>
#include <set>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
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
long long delta = inf;

constexpr bool sanity = true;
constexpr bool debug = false;
constexpr bool opt_delta = true;
constexpr bool only_main = true;

std::fstream err;

int all_vert;
int start, end;
int length;

long long max_len = 0;

int phases = 0, non_phase_steps = 0, local_relaxations = 0, non_local_relaxations = 0;

int numProcesses;
int myRank;

double elapsed_time;

template<class T, typename S>
vector<vector<T>> shareWithAll(const vector<S>& out_requests, const MPI_Datatype& mpi_type) {
	constexpr bool local_debug = false;

	if constexpr (local_debug) {
		err << "[share with all]" << std::endl;
	}
	// setup out request information
	vector<T> flat_out_requests;
	vector<int> out_request_addresses(numProcesses);

	vector<int> out_amounts(numProcesses, 0);
	vector<int> in_amounts(numProcesses, 0);
	int out_it = 0;
	for(int rank = 0; rank < numProcesses; rank++) {
		out_request_addresses[rank] = out_it;
		for(auto req: out_requests[rank]) {
			flat_out_requests.push_back(req);
			out_it++;
		}
		out_amounts[rank] = out_it - out_request_addresses[rank];
	}
	if constexpr (local_debug) {
		err << "[out information set up]" << std::endl;
	}

	// communicate request sizes
	MPI_Alltoall(
		out_amounts.data(), 
		1, MPI_INT, 
		in_amounts.data(), 
		1, MPI_INT, 
		MPI_COMM_WORLD);

	if constexpr (local_debug) {
		err << "[request sizes sent]" << std::endl;
	}

	// setup in request information
	vector<T> flat_in_requests;
	vector<int> in_request_addresses(numProcesses);
	int sum = 0;
	for(int rank = 0; rank < numProcesses; rank++) {
		in_request_addresses[rank] = sum;
		sum += in_amounts[rank];
	}
	flat_in_requests.resize(sum);

	if constexpr (local_debug) {
		err << "[in information set up]" << std::endl;
	}

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

	if constexpr (local_debug) {
		err << "[information sent]" << std::endl;

		err << "[amounts: ";
		for(auto x: in_amounts) {
			err << x << " ";
		}
		err << "]" << std::endl;

		err << "[in_request_addresses: ";
		for(auto x: in_request_addresses) {
			err << x << " ";
		}
		err << "]" << std::endl;

		err << "[";
		for(auto x: flat_in_requests) {
			err << x << " ";
		}
		err << "]" << std::endl;
	}

	vector<vector<T>> in_requests(numProcesses);
	for(int rank = 0; rank < numProcesses; rank++) {
		in_requests[rank].resize(in_amounts[rank]);
		for(int i = 0; i < in_amounts[rank]; i++) {
			in_requests[rank][i] = flat_in_requests[in_request_addresses[rank] + i];
		}
	}

	if constexpr (local_debug) {
		err << "[result calculated]" << std::endl;
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

set<pair<long long, int>> vertices_by_distance;
unordered_set<int> active;

LocalVector<vector<Target>> edges;
LocalVector<vector<Target>> short_edges;
LocalVector<long long> dist; ///< Local distances from 0
int settled = 0;
unordered_set<int> unsettled;
unordered_set<int> current_bucket;

inline void update_distance(int src, long long new_dist) {
	if constexpr (sanity) {
		auto it = vertices_by_distance.find({dist[src], src});
		if (it == vertices_by_distance.end()) {
			err << "<No vertex: " << src << " at distance: " << dist[src] << " while updating>" << std::endl; 
		}
	}

	vertices_by_distance.erase({dist[src], src});
	dist[src] = new_dist;
	vertices_by_distance.insert({new_dist, src});
}


void deltaEpochSetup(long long base) {
	if constexpr(debug) {
		err << " Starting delta epoch setup " << std::endl;
	}

	current_bucket = {};
	// Pick active vertices
	auto x = vertices_by_distance.begin();
	while(x != vertices_by_distance.end() && x->first < base + delta) {

		if constexpr(sanity) {
			if (unsettled.find(x->second) == unsettled.end()) {
				err << "<vertex " << x->second << " at dist " << x->first << " in vertex by distance was not unsettled>" << std::endl; 
			}
		}

		active.insert(x->second); 
		current_bucket.insert(x->second);
		x++;
	}
}

template<bool classification, bool pull>
bool deltaSingleStep(long long base) {
	if constexpr(debug) {
		err << "\tStarting delta single step with base: " << base << std::endl;
	}
	bool was_changed = false, global_changed = false;
	unordered_set<int> new_active;

	vector<unordered_map<int, long long>> best_update(numProcesses);

	if constexpr(debug) {
		err << "\tCurrently active: ";
		for(auto src: active) {
			err << src << " ";
		}
		err << std::endl;
	}

	for(auto src: active) {
		if constexpr (debug) {
			err << "\t\tCalculating vertex: " << src << std::endl;
		}
		auto edges_begin = edges[src].begin();
		auto edges_end = edges[src].end();
		if constexpr(classification) {
			edges_begin = short_edges[src].begin();
			edges_end = std::lower_bound(short_edges[src].begin(), short_edges[src].end(), Target::lower(base + delta - dist[src]));
		}
		if constexpr (debug) {
			if (edges_begin == edges_end) {
				err << "\t\tNo short edges" << std::endl;
			}
		}
		for(auto it = edges_begin; it != edges_end; it++) {
			auto& target = *it;
			if constexpr (debug) {
				err << "\t\tEdge to " << target.dest << " of length " << target.length << std::endl;
			}
			long long new_dist = dist[src] + target.length;
			if (is_local(target.dest)) {
				local_relaxations++;
				if (new_dist < dist[target.dest]) {

					if constexpr (classification) {
						if constexpr (sanity) {
							if (new_dist >= base + delta) {
								err << "<<First assumption wrong>>" << std::endl;
							}
						}
						if (dist[target.dest] >= base + delta) {
							current_bucket.insert(target.dest);
						}
						new_active.insert(target.dest);
						was_changed = true;
					} else if (new_dist < base + delta) {
						if (dist[target.dest] >= base + delta) {
							current_bucket.insert(target.dest);
						}
						new_active.insert(target.dest);
						was_changed = true;
					}
					update_distance(target.dest, new_dist);
				}
			} else {
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

	auto in_vertex = shareWithAll<int>(out_vertex, MPI_INT);
	auto in_dist = shareWithAll<long long>(out_dist, MPI_LONG_LONG);

	for(int rank = 0; rank < numProcesses; rank++) {
		for(int i = 0; i < in_vertex[rank].size(); i++) {
			auto dest = in_vertex[rank][i];
			auto new_dist = in_dist[rank][i];
			non_local_relaxations++;
			if (new_dist < dist[dest]) {
				// Possibly not useful if anymore
				if (new_dist < base + delta) {
					if (dist[dest] >= base + delta) {
						current_bucket.insert(dest);
					}
					new_active.insert(dest); 
					was_changed = true;
				}
				update_distance(dest, new_dist);
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

template<bool pull>
void deltaLongPhase(int base) {
	if constexpr (debug) {
		err << " Delta Long Phase" << std::endl;
	}
	if constexpr (!pull) {
		vector<unordered_map<int, long long>> best_update(numProcesses);

		// Handle edges going out of the current bracket forward.
		for(auto src: current_bucket) {
			auto edges_begin = std::lower_bound(edges[src].begin(), edges[src].end(), Target::lower(base + delta - dist[src]));
			auto edges_end = edges[src].end();
			for(auto it = edges_begin; it != edges_end; it++) {
				auto& target = *it;
				auto new_dist = target.length + dist[src];
				if (is_local(target.dest)) {
					local_relaxations++;
					if (new_dist < dist[target.dest]) {
						if constexpr (sanity) {
							if (new_dist < base + delta) {
								err << "<<Second assumption wrong>>" << std::endl;
							}
						}

						active.insert(target.dest);
						update_distance(target.dest, new_dist);
					}
				} else {
					auto mit = best_update[target.destRank].find(target.dest);
					auto best = new_dist;
					if (mit != best_update[target.destRank].end()) {
						best = min(mit->second, new_dist);
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

		auto in_vertex = shareWithAll<int>(out_vertex, MPI_INT);
		auto in_dist = shareWithAll<long long>(out_dist, MPI_LONG_LONG);

		for(int rank = 0; rank < numProcesses; rank++) {
			for(int i = 0; i < in_vertex[rank].size(); i++) {
				auto dest = in_vertex[rank][i];
				auto new_dist = in_dist[rank][i];
				non_local_relaxations++;
				if (new_dist < dist[dest]) {
					active.insert(dest); 
					update_distance(dest, new_dist);
				}	
			} 
		}
	} else {
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
					if (dist[target.dest] + target.length < dist[src]) {
						update_distance(src, dist[target.dest] + target.length);
						active.insert(src);
					}
				} else if (target.length + base < dist[src]) {
					out_requests[target.destRank].insert(target.dest);
					attributed_requests[target.destRank].emplace_back(src, target.length, target.dest);
				}
			}
		}
		auto in_requests = shareWithAll<int>(out_requests, MPI_INT);

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
		auto in_vertices = shareWithAll<int>(out_vertices, MPI_INT);
		auto in_lengths = shareWithAll<long long>(out_lengths, MPI_LONG_LONG);

		for(auto rank = 0; rank < numProcesses; rank++) {
			unordered_map<int, long long> partial_outside;
			for(int i = 0; i < in_vertices[rank].size(); i++) {
				partial_outside[in_vertices[rank][i]] = in_lengths[rank][i];
			}
			for(auto [src, len, dest]: attributed_requests[rank]) {
				auto it = partial_outside.find(dest);
				non_local_relaxations++;
				if (it != partial_outside.end()) {
					auto new_dist = len + it->second;
					if (new_dist < dist[src]) {
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
	if constexpr(debug) {
		err << " Delta epilogue" << std::endl;
	}
	active = {};

	for(auto src: current_bucket) {
		unsettled.erase(src);
		settled++;
		vertices_by_distance.erase({dist[src], src});
	}

	if constexpr (classification) {
		deltaLongPhase<pull>(base);
	}
}

long long calc_min_dist() {
	long long min_dist = inf, global_min_dist = inf;
	if (!vertices_by_distance.empty()) {
		min_dist = vertices_by_distance.begin()->first;
	}
	MPI_Allreduce(
		&min_dist, 
		&global_min_dist, 
		1, 
		MPI_LONG_LONG, 
		MPI_MIN, 
		MPI_COMM_WORLD
	);

	return global_min_dist;
}

template <bool classification, bool pull>
bool deltaEpoch() {
	if constexpr (debug) {
		err << "Starting new epoch nr: " << phases << std::endl;
	}
	static_assert(!pull || classification, "pull model only available if classification is enabled");

	long long base = calc_min_dist();

	if (base == inf) return false;

	deltaEpochSetup(base);

	non_phase_steps++;
	while(deltaSingleStep<classification, pull>(base)) {
		non_phase_steps++;
	}

	deltaEpochEpilogue<classification, pull>(base);

	return true;
}

void runBellmanFord() {
	auto base = calc_min_dist();
	delta = inf - base;	

	non_phase_steps++;
	while(deltaEpoch<false, false>()) {
		non_phase_steps++;
	}
}

template<bool classification, bool pull, bool hybridize>
void runDelta() {
	phases++;
	while(deltaEpoch<classification, pull>()) {
		phases++;
		if constexpr(hybridize) {
			int all_settled;
			MPI_Allreduce(
				&settled, 
				&all_settled, 
				1, 
				MPI_INT, 
				MPI_SUM, 
				MPI_COMM_WORLD
			);
			if (all_settled * 10 >= all_vert) {
				runBellmanFord();
				break;
			}
		}
	}
}

void setup() {
	dist->resize(length);
	ends.resize(numProcesses);
	// Collect common start/end of vertices of each process
	for(int src = start; src <= end; src++) {
		unsettled.insert(src);
	}
	if (myRank == 0) {
		active = {0};
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

	if constexpr (opt_delta) {
		// Calculate optimal delta
		long long local_max = max_len;	
		MPI_Allreduce(
			&local_max,
			&max_len,
			1, MPI_LONG_LONG,
			MPI_MAX,
			MPI_COMM_WORLD
		);
		delta = std::max(10LL, max_len / 10);
		if (myRank == 0 || !only_main) {
			err << "Delta " << delta << " was chosen" << std::endl;
		}
	}

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
	if constexpr (debug) {
		err << "Reading input" << std::endl;
	}
	std::fstream in(file, std::ios_base::in);

	in >> all_vert >> start >> end;
	length = end - start + 1;
	edges->resize(length);
	short_edges->resize(length);
	int x, y;
	long long len;
	while(in >> x) {
		in >> y >> len;
		max_len = std::max(max_len, len);
		if (is_local(x)) {
			edges[x].emplace_back(y, len);
		}
		if (is_local(y)) {
			edges[y].emplace_back(x, len);
		}
	}
	for(int src = start; src <= end; src++) {
		sort(edges[src].begin(), edges[src].end());
	}
	in.close();
	if constexpr (debug) {
		err << "Finished reading input" << std::endl;
	}
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
	out.close();
}

void write_info() {
	int local_relaxations_sum;
	MPI_Allreduce(
		&local_relaxations, 
		&local_relaxations_sum, 
		1, MPI_INT, 
		MPI_SUM, 
		MPI_COMM_WORLD
	);

	int non_local_relaxations_sum;
	MPI_Allreduce(
		&non_local_relaxations, 
		&non_local_relaxations_sum, 
		1, MPI_INT, 
		MPI_SUM, 
		MPI_COMM_WORLD
	);

	if (debug || (!only_main || myRank == 0)) {
		err << "Number of phases: " << phases << std::endl;
		err << "Number of non phase steps: " << non_phase_steps << std::endl;
		err << "Local relaxations in rank " << myRank << ": " << local_relaxations << std::endl;
		err << "Local relaxations in summary: " << local_relaxations_sum << std::endl;
		err << "Non Local relaxations in rank " << myRank << ": " << non_local_relaxations << std::endl;
		err << "Non Local relaxations in summary: " << non_local_relaxations_sum << std::endl;
		err << "Elapsed time: " << elapsed_time << "s" << std::endl;
	}
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		cerr << "Usage: " << string(argv[0]) << " input_file output_file"; 
		return 1;
	}

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	if (debug || (!only_main || myRank == 0)) {
		const std::regex ext("[.].*$");
		const std::regex sep("[/]");

		string in_path = argv[1];
		string changed_extension = std::regex_replace(in_path, ext, ".err");
		string err_path = "error/" + std::regex_replace(changed_extension, sep, "--");

		err.open(err_path, std::ios_base::out);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	read(argv[1]);
	double start_time = MPI_Wtime();

	setup();

	runDelta<true, true, false>();

	double end_time = MPI_Wtime();

	elapsed_time = end_time - start_time;

	write_info();
	write_out(argv[2]);

	if (debug || (!only_main || myRank == 0)) {
		err.close();
	}
}
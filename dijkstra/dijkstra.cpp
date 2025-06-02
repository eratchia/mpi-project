#include <iostream>
#include <vector>
#include <queue>

using std::pair;
using std::vector;
using std::priority_queue;
using std::cin;
using std::cerr;
using std::cout;

int start, end;
vector<vector<std::pair<int, long long>>> edges;

inline int to_local(const int x) {
	return x - start;
}

inline int to_global(const int x) {
	return x + start;
}

inline bool is_local(const int x) {
	return x >= start && x < end;
}

int main() {
	std::ios_base::sync_with_stdio(0);
	cin.tie(0);

	int n;
	cin >> n;
	edges.resize(n);
	cin >> start >> end;

	int x, y;
	long long len;
	while (cin >> x) {
		cin >> y >> len;
		if (is_local(x)) {
			edges[to_local(x)].emplace_back(y, len);
		} if (is_local(y)) {
			edges[to_local(y)].emplace_back(x, len);
		}
	}
	vector<long long> dist(end - start, -1);
	vector<bool> vis(end - start, false);
	if (is_local(0)) {
		dist[to_local(0)] = 0;
	}
	priority_queue<pair<long long, int>, vector<pair<long long, int>>, std::greater<pair<long long, int>>> q;
	q.emplace(0, 0);
	while(!q.empty()) {
		auto [d, x] = q.top();
		cerr << d << " " << x << "\n";
		q.pop();
		if (vis[x]) continue;
		vis[x] = true;
		for(auto& [y, add]: edges[x]) {
			if (dist[y] == -1 || d + add < dist[y]) {
				dist[y] = d + add;
				q.emplace(d + add, y);
			}
		}
	}
	for (int i = start; i < end; i++) {
		cout << dist[to_local(i)] << "\n";
	}
}
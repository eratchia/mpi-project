#include <iostream>
#include <string>
#include <fstream>
#include <regex>
#include <sys/stat.h>

using namespace std;

int nodes, processes, vertices;
int base, bigger;
int last_bigger;

int whichProcess(int vertex) {
	if (vertex < last_bigger) return vertex / (base + 1);
	return bigger + (vertex - last_bigger) / base;
}

int main(int argc, char* argv[]) {
	ios_base::sync_with_stdio(0);
	srand(time(0));
	if (argc != 4) {
		cerr << "Usage: " << string(argv[0]) << " nodes input_file test_dir"; 
		return 1;
	}

	string input_file = argv[2];
	string test_dir = argv[3];
	const regex getname("/[^/]*[.]");

	smatch matched;
	regex_search(input_file, matched, getname);
	string name(matched[0]);


	string out_dir = test_dir + "/" + name;
	mkdir(out_dir.c_str(), 0777);

	nodes = stoi(argv[1]);
	processes = 24 * nodes;
	vertices = nodes * 1000000;
	bigger = vertices % processes;
	base = vertices / processes;
	last_bigger = bigger * (base + 1);

	fstream in(input_file, ios_base::in);
	vector<fstream> outs;

	for(int p = 0; p < processes; p++) {
		outs.emplace_back();
	}

	for(int p = 0; p < processes; p++) {
		string file = out_dir + "/" + to_string(p) + ".in";
		outs[p].open(file, ios_base::out);
		outs[p] << vertices << " " << base * (p - 1) + min(bigger, p - 1) << " " << base * p + min(bigger, p) << "\n";
	}

	int x, y, len;
	while(cin >> x) {
		cin >> y;
		len = rand() % 256;
		int x_p = whichProcess(x);
		int y_p = whichProcess(y);
		outs[x_p] << x << " " << y << " " << len << "\n";
		if (y_p != x_p) {
			outs[y_p] << x << " " << y << " " << len << "\n";
		}
	}
	in.close();
	for(int p = 0; p < processes; p++) {
		outs[p].close();
	}
}
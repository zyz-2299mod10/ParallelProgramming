#include <iostream>
#include <random>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

using namespace std;

// shared variables
long long int number_in_circle = 0;
long long int number_of_tosses;
long long int chunk;
int threads_num;
mutex mtx;

void estimate_pi(int threads_index) {
    long long int start = chunk * threads_index;
    long long int end = (threads_index == threads_num - 1) ? number_of_tosses : start + chunk;

    unsigned seed = threads_index;
    long long int local_count = 0;

    for (long long int i = start; i < end; ++i) {
        double x = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;
        double y = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;
        double distance_squared = x * x + y * y;

        if (distance_squared <= 1) {
            ++local_count;
        }
    }

    // Only lock once at the end to update shared variable
    lock_guard<mutex> guard(mtx);
    number_in_circle += local_count;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <number of threads> <number of tosses>" << endl;
        return EXIT_FAILURE;
    }

    threads_num = stoi(argv[1]);
    number_of_tosses = stoll(argv[2]);
    chunk = number_of_tosses / threads_num;

    // auto start = chrono::high_resolution_clock::now();

    vector<thread> threads(threads_num);
    for (int threads_index = 0; threads_index < threads_num; ++threads_index) {
        threads[threads_index] = thread(estimate_pi, threads_index);
    }

    for (auto& t : threads) {
        t.join();
    }

    double pi_estimate = 4 * static_cast<double>(number_in_circle) / number_of_tosses;
    cout << pi_estimate << endl;

    // auto end = chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed = end - start;
    // cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}
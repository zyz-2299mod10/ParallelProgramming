#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>
#include <mutex>
#include <thread>
#include <CycleTimer.h>
#include <type.h>
#include <omp.h>

using namespace block;
using SolutionBoard = std::array<std::array<uint_fast8_t, 7>, 7>;  // 單個 7x7 棋盤的解
using DailySolutions = std::vector<SolutionBoard>;                 // 一天的所有解
using MonthlySolutions = std::vector<DailySolutions>;              // 每月的所有天解
using YearlySolutions = std::vector<MonthlySolutions>;             // 一整年的所有解

YearlySolutions solutions(12, MonthlySolutions(31)); // 12 個月，每個月都先開31天
std::mutex solutions_mutex;

const std::span<const uint_fast8_t[4][4]> Pieces[8] = {
    Hexomino,    L_Pentomino, N_Pentomino, P_Pentomino,
    U_Pentomino, V_Pentomino, Y_Pentomino, Z_Pentomino,
};

bool can_place(const uint_fast8_t Board[10][10], const uint_fast8_t shape[4][4], const uint_fast8_t x, const uint_fast8_t y) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0 && Board[y + j][x + i] != 0) {
                return false;
            }
        }
    }
    return true;
}

void place(uint_fast8_t Board[10][10], const uint_fast8_t shape[4][4], const uint_fast8_t x, const uint_fast8_t y) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0) {
                Board[y + j][x + i] = shape[j][i];
            }
        }
    }
}

void unplace(uint_fast8_t Board[10][10], const uint_fast8_t shape[4][4], const uint_fast8_t x, const uint_fast8_t y) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0) {
                Board[y + j][x + i] = 0;
            }
        }
    }
}

bool valid(const uint_fast8_t Board[10][10]) {
    uint_fast8_t month_count = 0;
    for (uint_fast8_t i = 0; i < 2; ++i) {
        for (uint_fast8_t j = 0; j < 7; ++j) {
            if (Board[i][j] == 0) {
                ++month_count;
                if (month_count > 1) {
                    return false;
                }
            }
        }
    }
    if (month_count == 0) {
        return false;
    }
    return true;
}

uint_fast8_t get_month(const uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 2; ++i) {
        for (uint_fast8_t j = 0; j < 7; ++j) {
            if (Board[i][j] == 0) {
                return i * 6 + j + 1;
            }
        }
    }
    return 0;
}

uint_fast8_t get_day(const uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 2; i < 7; ++i) {
        for (uint_fast8_t j = 0; j < 7; ++j) {
            if (Board[i][j] == 0) {
                return (i - 2) * 7 + j + 1;
            }
        }
    }
    return 0;
}

void backtrack(uint_fast8_t Board[10][10], uint_fast8_t placed, std::span<const std::span<const uint_fast8_t[4][4]>> pieces) {
    if (placed == 8) {
        if (!valid(Board)) return;

        uint16_t month = get_month(Board);
        uint16_t day = get_day(Board);

        SolutionBoard currentSolution;
        for (uint_fast8_t i = 0; i < 7; ++i) {
            for (uint_fast8_t j = 0; j < 7; ++j) {
                currentSolution[i][j] = Board[i][j];
            }
        }

        // For verification
        std::ofstream fout("./result/" + std::to_string(month) + '_' + std::to_string(day) + ".txt", std::ios::app);
        for (uint_fast8_t i = 0; i < 7; ++i) {
            for (uint_fast8_t j = 0; j < 7; ++j) {
                fout << static_cast<uint16_t>(currentSolution[i][j]);
            }
            fout << '\n';
        }
        fout << '\n';

        std::lock_guard<std::mutex> lock(solutions_mutex);
        solutions[month - 1][day - 1].push_back(currentSolution);
        return;
    }

    if (placed + pieces.size() < 8) return;

    uint_fast8_t LocalBoard[10][10];
    memcpy(LocalBoard, Board, 10 * 10 * sizeof(uint_fast8_t)); // Explicitly specify size

    for (uint_fast8_t i = 0; i < pieces.size(); ++i) {
        const auto piece = pieces[i];
        for (const auto shape : piece) {
            // bool placed_piece = false;
            for (uint_fast8_t x = 0; x < 7; ++x) {
                for (uint_fast8_t y = 0; y < 7; ++y) {
                    if (can_place(LocalBoard, shape, x, y)) {
                        place(LocalBoard, shape, x, y);
                        #pragma omp task firstprivate(LocalBoard)
                        {
                            backtrack(LocalBoard, placed + 1, pieces.subspan(i + 1));
                        }
                        unplace(LocalBoard, shape, x, y);
                        // placed_piece = true;
                    }
                }
            }
            // if (placed_piece) break;
        }
    }
    #pragma omp taskwait
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    int thread_count = -1;
    if (argc == 2)
    {
        thread_count = atoi(argv[1]);
    }

    printf("----------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    if (thread_count > 0)
    {
        thread_count = std::min(thread_count, omp_get_max_threads());
        printf("Running with %d threads\n", thread_count);
    }
    printf("----------------------------------------------------------\n");

    if (thread_count <= -1){
        int max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
    }
    else{
        omp_set_num_threads(thread_count);
    }

    uint_fast8_t InitialBoard[10][10] = {
        {0, 0, 0, 0, 0, 0, 9, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 0, 0, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 0, 9, 9, 9},
        {0, 0, 0, 0, 0, 0, 0, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 0, 9, 9, 9},
        {0, 0, 0, 9, 9, 9, 9, 9, 9, 9}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
        {9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
    };
    
    double start_time = currentSeconds();
    #pragma omp parallel
    {
        #pragma omp single
        {
            backtrack(InitialBoard, 0, Pieces);
        }
    }
    double end_time = currentSeconds();
    double ElapsedTime = end_time - start_time;
    std::cout << "Elapsed Time: " << ElapsedTime << " (s)" << std::endl;

    return 0;
}
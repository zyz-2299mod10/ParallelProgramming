#include <array>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>
#include <mutex>
#include <thread>
#include <algorithm>
#include <type.h>

// 定義解決方案的類型
using namespace block;
using SolutionBoard = std::array<std::array<uint_fast8_t, 7>, 7>;  // 單個 7x7 的解
using DailySolutions = std::vector<SolutionBoard>;                 // 一天的所有解
using MonthlySolutions = std::vector<DailySolutions>;              // 每月的所有天解
using YearlySolutions = std::vector<MonthlySolutions>;             // 一整年的所有解

YearlySolutions one_thread_solutions(12, MonthlySolutions(31)); // 12 個月，每個月先開31天
std::vector<YearlySolutions> solutions(8, one_thread_solutions);

const std::span<const uint_fast8_t[4][4]> Pieces[8] = {
    block::U_Pentomino, block::Hexomino,    block::N_Pentomino, block::P_Pentomino,
    block::L_Pentomino, block::V_Pentomino, block::Y_Pentomino, block::Z_Pentomino,
};


inline bool can_place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
               const uint_fast8_t y, const uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0 && Board[y + j][x + i] != 0) {
                return false;
            }
        }
    }
    return true;
}


inline void place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
           const uint_fast8_t y, uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0) {
                Board[y + j][x + i] = shape[j][i];
            }
        }
    }
}


inline void unplace(const uint_fast8_t shape[4][4], const uint_fast8_t x,
             const uint_fast8_t y, uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0) {
                Board[y + j][x + i] = 0;
            }
        }
    }
}


inline bool valid(const uint_fast8_t Board[10][10]) {
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


inline uint_fast8_t get_month(const uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 2; ++i) {
        for (uint_fast8_t j = 0; j < 7; ++j) {
            if (Board[i][j] == 0) {
                return i * 6 + j + 1;
            }
        }
    }
    return 0;
}


inline uint_fast8_t get_day(const uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 2; i < 7; ++i) {
        for (uint_fast8_t j = 0; j < 7; ++j) {
            if (Board[i][j] == 0) {
                return (i - 2) * 7 + j + 1;
            }
        }
    }
    return 0;
}


inline void backtrack(int shape_idx,
							 uint_fast8_t placed,
               std::span<const std::span<const uint_fast8_t[4][4]>> pieces,
               uint_fast8_t Board[10][10]) {
    if (placed == 8) {
        if (!valid(Board)) {
            return;
        }
        const uint16_t month = get_month(Board) - 1; // 當作 array 的 month index
        const uint16_t day = get_day(Board) - 1;     // 當作 array 的 day index

        SolutionBoard currentSolution;
        for (uint_fast8_t i = 0; i < 7; ++i) {
            for (uint_fast8_t j = 0; j < 7; ++j) {
                currentSolution[i][j] = Board[i][j];
            }
        }

        solutions[shape_idx][month][day].push_back(currentSolution);
        

        return;
    }
    if (placed + pieces.size() < 8) {
        return;
    }
    for (uint_fast8_t i = 0; i < pieces.size(); ++i) {
        const auto piece = pieces[i];
        for (const auto& shape : piece) {
            for (uint_fast8_t x = 0; x < 7; ++x) {
                for (uint_fast8_t y = 0; y < 7; ++y) {
                    if (can_place(shape, x, y, Board)) {
                        place(shape, x, y, Board);
                        backtrack(shape_idx, placed + 1, pieces.subspan(i + 1), Board);
                        unplace(shape, x, y, Board);
                    }
                }
            }
        }
    }
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    uint_fast8_t initial_Board[10][10] = {
        {0, 0, 0, 0, 0, 0, 9, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 0, 0, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 0, 9, 9, 9},
        {0, 0, 0, 0, 0, 0, 0, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 0, 9, 9, 9},
        {0, 0, 0, 9, 9, 9, 9, 9, 9, 9}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
        {9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
    };

    std::vector<std::thread> threads;

    
    const auto& first_piece = Pieces[0];

    // 每個不同的旋轉方式創建一個thread
    for (size_t shape_idx = 0; shape_idx < first_piece.size(); ++shape_idx) {
        uint_fast8_t first_shape_copy[4][4];
        std::copy(&first_piece[shape_idx][0][0], &first_piece[shape_idx][0][0] + 16, &first_shape_copy[0][0]);

        threads.emplace_back([shape_idx, first_shape_copy, &initial_Board, &solutions]() {
            for (uint_fast8_t x = 0; x < 7; ++x) {
                for (uint_fast8_t y = 0; y < 7; ++y) {
                    
                    if (can_place(first_shape_copy, x, y, initial_Board)) {
                        
                        uint_fast8_t new_Board[10][10];
                        std::copy(&initial_Board[0][0], &initial_Board[0][0] + 100, &new_Board[0][0]);

                        place(first_shape_copy, x, y, new_Board);

                        std::span<const std::span<const uint_fast8_t[4][4]>> remaining_pieces(Pieces + 1, 7);


                        backtrack(shape_idx, 1, remaining_pieces, new_Board);
                    }
                }
            }
        });
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    
    
    for (uint_fast8_t month = 0; month < 12; ++month) {
         for (uint_fast8_t day = 0; day < 31; ++day) {
		std::ofstream fout("sol/" + std::to_string(month + 1) + '_' + std::to_string(day + 1) + "_v3.txt",
                                   std::ios::app);
		for(int idx=0;idx<8;idx++){
            if (!solutions[idx][month][day].empty()) {
                
                for (const auto& sol : solutions[idx][month][day]) {
                    for (uint_fast8_t i = 0; i < 7; ++i) {
                        for (uint_fast8_t j = 0; j < 7; ++j) {
                            fout << static_cast<uint16_t>(sol[i][j]);
                        }
                        fout << '\n';
                    }
                    fout << '\n';
                }
            }
		}
         }
    }

    return 0;
}
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
using SolutionBoard = std::array<std::array<uint_fast8_t, 7>, 7>;  // 單個 7x7 棋盤的解
using DailySolutions = std::vector<SolutionBoard>;                 // 一天的所有解
using MonthlySolutions = std::vector<DailySolutions>;              // 每月的所有天解
using YearlySolutions = std::vector<MonthlySolutions>;             // 一整年的所有解

YearlySolutions solutions(12, MonthlySolutions(31)); // 12 個月，每個月先開31天
std::mutex solutions_mutex;

const std::span<const uint_fast8_t[4][4]> Pieces[8] = {
    block::U_Pentomino, block::Hexomino,    block::N_Pentomino, block::P_Pentomino,
    block::L_Pentomino, block::V_Pentomino, block::Y_Pentomino, block::Z_Pentomino,
};

// 檢查是否可以放置形狀
bool can_place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
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

// 放置形狀
void place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
           const uint_fast8_t y, uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0) {
                Board[y + j][x + i] = shape[j][i];
            }
        }
    }
}

// 拆除形狀
void unplace(const uint_fast8_t shape[4][4], const uint_fast8_t x,
             const uint_fast8_t y, uint_fast8_t Board[10][10]) {
    for (uint_fast8_t i = 0; i < 4; ++i) {
        for (uint_fast8_t j = 0; j < 4; ++j) {
            if (shape[j][i] != 0) {
                Board[y + j][x + i] = 0;
            }
        }
    }
}

// 驗證當前board狀態是否有效
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

// 獲取月份
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

// 獲取日期
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

// 遞回主函數
void backtrack(uint_fast8_t placed,
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

        // 保護共享資源
        {
            std::lock_guard<std::mutex> lock(solutions_mutex);
            solutions[month][day].push_back(currentSolution);
        }

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
                        backtrack(placed + 1, pieces.subspan(i + 1), Board);
                        unplace(shape, x, y, Board);
                    }
                }
            }
        }
    }
}

// 生成初始放置並分配給不同的執行緒
void thread_worker(const uint_fast8_t first_shape[4][4],
                  std::span<const std::span<const uint_fast8_t[4][4]>> remaining_pieces,
                  uint_fast8_t initial_Board[10][10]) {
    // 放第一個方塊
    place(first_shape, 0, 0, initial_Board); // 先放在 (0,0)

    backtrack(1, remaining_pieces, initial_Board);
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

    // 選擇第一個方塊的所有旋轉方式
    const auto& first_piece = Pieces[0]; // 假設第一個方塊是 Hexomino

    // 為每個不同的旋轉方式創建一個執行緒
    for (size_t shape_idx = 0; shape_idx < first_piece.size(); ++shape_idx) {
        // 為避免 lambda 捕獲錯誤，將 first_shape 複製一份
        uint_fast8_t first_shape_copy[4][4];
        std::copy(&first_piece[shape_idx][0][0], &first_piece[shape_idx][0][0] + 16, &first_shape_copy[0][0]);

        // 創建一個執行緒處理這種旋轉方式
        threads.emplace_back([first_shape_copy, &initial_Board, &solutions, &solutions_mutex]() {
            for (uint_fast8_t x = 0; x < 7; ++x) {
                for (uint_fast8_t y = 0; y < 7; ++y) {
                    // 檢查是否可以放置
                    if (can_place(first_shape_copy, x, y, initial_Board)) {
                        // 複製板狀態
                        uint_fast8_t new_Board[10][10];
                        std::copy(&initial_Board[0][0], &initial_Board[0][0] + 100, &new_Board[0][0]);

                        // 放置形狀
                        place(first_shape_copy, x, y, new_Board);

                        // 定義剩餘的塊（從下一個塊開始）
                        std::span<const std::span<const uint_fast8_t[4][4]>> remaining_pieces(Pieces + 1, 7);

                        // 開始回溯
                        backtrack(1, remaining_pieces, new_Board);
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
    int month = 0;
    int day = 0;
    // 將解決方案寫入檔案
    for (uint_fast8_t month = 0; month < 12; ++month) {
        for (uint_fast8_t day = 0; day < 31; ++day) {
            if (!solutions[month][day].empty()) {
                std::ofstream fout(std::to_string(month + 1) + '_' + std::to_string(day + 1) + "_v2.txt",
                                   std::ios::app);
                for (const auto& sol : solutions[month][day]) {
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

    return 0;
}

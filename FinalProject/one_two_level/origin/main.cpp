#include <array>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>
#include <mutex>
#include <thread>

#include "type.h"
#include "CycleTimer.h"

using namespace block;
using SolutionBoard = std::array<std::array<uint_fast8_t, 7>, 7>;  // 單個 7x7 棋盤的解
using DailySolutions = std::vector<SolutionBoard>;                 // 一天的所有解
using MonthlySolutions = std::vector<DailySolutions>;              // 每月的所有天解
using YearlySolutions = std::vector<MonthlySolutions>;             // 一整年的所有解

YearlySolutions solutions(12, MonthlySolutions(31)); // 12 個月，每個月都先開31天


uint_fast8_t Board[10][10] = {
      {0, 0, 0, 0, 0, 0, 9, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 9, 9, 9, 9},
      {0, 0, 0, 0, 0, 0, 0, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 0, 9, 9, 9},
      {0, 0, 0, 0, 0, 0, 0, 9, 9, 9}, {0, 0, 0, 0, 0, 0, 0, 9, 9, 9},
      {0, 0, 0, 9, 9, 9, 9, 9, 9, 9}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
      {9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
  };

const std::span<const uint_fast8_t[4][4]> Pieces[8] = {
    Hexomino,    L_Pentomino, N_Pentomino, P_Pentomino,
    U_Pentomino, V_Pentomino, Y_Pentomino, Z_Pentomino,
};

bool can_place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
               const uint_fast8_t y) {
  for (uint_fast8_t i = 0; i < 4; ++i) {
    for (uint_fast8_t j = 0; j < 4; ++j) {
      if (shape[j][i] != 0 && Board[y + j][x + i] != 0) {
        return false;
      }
    }
  }
  return true;
}

void place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
           const uint_fast8_t y) {
  for (uint_fast8_t i = 0; i < 4; ++i) {
    for (uint_fast8_t j = 0; j < 4; ++j) {
      if (shape[j][i] != 0) {
        Board[y + j][x + i] = shape[j][i];
      }
    }
  }
}

void unplace(const uint_fast8_t shape[4][4], const uint_fast8_t x,
             const uint_fast8_t y) {
  for (uint_fast8_t i = 0; i < 4; ++i) {
    for (uint_fast8_t j = 0; j < 4; ++j) {
      if (shape[j][i] != 0) {
        Board[y + j][x + i] = 0;
      }
    }
  }
}

bool valid() {
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

uint_fast8_t get_month() {
  for (uint_fast8_t i = 0; i < 2; ++i) {
    for (uint_fast8_t j = 0; j < 7; ++j) {
      if (Board[i][j] == 0) {
        return i * 6 + j + 1;
      }
    }
  }
  return 0;
}

uint_fast8_t get_day() {
  for (uint_fast8_t i = 2; i < 7; ++i) {
    for (uint_fast8_t j = 0; j < 7; ++j) {
      if (Board[i][j] == 0) {
        return (i - 2) * 7 + j + 1;
      }
    }
  }
  return 0;
}

void backtrack(uint_fast8_t placed,
               std::span<const std::span<const uint_fast8_t[4][4]>> pieces) {
  using namespace std::literals;
  if (placed == 8) {
    if (!valid()) {
      return;
    }
    const uint16_t month = get_month() - 1; // 當作 array 的 month index
    const uint16_t day = get_day() - 1; // //當作 array 的 day index

    SolutionBoard currentSolution;
    for (uint_fast8_t i = 0; i < 7; ++i) {
      for (uint_fast8_t j = 0; j < 7; ++j) {
        currentSolution[i][j] = Board[i][j];
      }
    }

    // For checking answer
    std::ofstream fout("./answer/" + std::to_string(month + 1) + '_' + std::to_string(day + 1) + ".txt"s, std::ios::app);
    for (uint_fast8_t i = 0; i < 7; ++i) {
      for (uint_fast8_t j = 0; j < 7; ++j) {
        fout << static_cast<uint16_t>(currentSolution[i][j]);
      }
      fout << '\n';
    }
    fout << '\n';

    solutions[month][day].push_back(currentSolution);

    return;
  }
  if (placed + pieces.size() < 8) {
    return;
  }
  for (uint_fast8_t i = 0; i < pieces.size(); ++i) {
    const auto piece = pieces[i];
    for (const auto shape : piece) {
      for (uint_fast8_t x = 0; x < 7; ++x) {
        for (uint_fast8_t y = 0; y < 7; ++y) {
          if (can_place(shape, x, y)) {
            place(shape, x, y);
            backtrack(placed + 1, pieces.subspan(i + 1));
            unplace(shape, x, y);
          }
        }
      }
    }
  }
}

int main() {
  std::ios::sync_with_stdio(false);

  double start_time = CycleTimer::currentSeconds();
  
  backtrack(0, Pieces);

  double end_time = CycleTimer::currentSeconds();
  double ElapsedTime = end_time - start_time;
  std::cout << "Elapsed Time: " << ElapsedTime << " (s)" << std::endl;
}
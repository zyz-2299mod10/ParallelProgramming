#include <array>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>
#include <thread>
#include <omp.h>

#include "type.h"
#include "CycleTimer.h"

using namespace block;
using SolutionBoard = std::array<std::array<uint_fast8_t, 7>, 7>;
using DailySolutions = std::vector<SolutionBoard>;
using MonthlySolutions = std::vector<DailySolutions>;
using YearlySolutions = std::vector<MonthlySolutions>;

YearlySolutions solutions(12, MonthlySolutions(31));

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
               const uint_fast8_t y, uint_fast8_t localBoard[10][10]) {
  for (uint_fast8_t i = 0; i < 4; ++i) {
    for (uint_fast8_t j = 0; j < 4; ++j) {
      if (shape[j][i] != 0 && localBoard[y + j][x + i] != 0) {
        return false;
      }
    }
  }
  return true;
}

void place(const uint_fast8_t shape[4][4], const uint_fast8_t x,
           const uint_fast8_t y, uint_fast8_t localBoard[10][10]) {
  for (uint_fast8_t i = 0; i < 4; ++i) {
    for (uint_fast8_t j = 0; j < 4; ++j) {
      if (shape[j][i] != 0) {
        localBoard[y + j][x + i] = shape[j][i];
      }
    }
  }
}

void unplace(const uint_fast8_t shape[4][4], const uint_fast8_t x,
             const uint_fast8_t y, uint_fast8_t localBoard[10][10]) {
  for (uint_fast8_t i = 0; i < 4; ++i) {
    for (uint_fast8_t j = 0; j < 4; ++j) {
      if (shape[j][i] != 0) {
        localBoard[y + j][x + i] = 0;
      }
    }
  }
}

bool valid(uint_fast8_t localBoard[10][10]) {
  uint_fast8_t month_count = 0;
  for (uint_fast8_t i = 0; i < 2; ++i) {
    for (uint_fast8_t j = 0; j < 7; ++j) {
      if (localBoard[i][j] == 0) {
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

uint_fast8_t get_month(uint_fast8_t localBoard[10][10]) {
  for (uint_fast8_t i = 0; i < 2; ++i) {
    for (uint_fast8_t j = 0; j < 7; ++j) {
      if (localBoard[i][j] == 0) {
        return i * 6 + j + 1;
      }
    }
  }
  return 0;
}

uint_fast8_t get_day(uint_fast8_t localBoard[10][10]) {
  for (uint_fast8_t i = 2; i < 7; ++i) {
    for (uint_fast8_t j = 0; j < 7; ++j) {
      if (localBoard[i][j] == 0) {
        return (i - 2) * 7 + j + 1;
      }
    }
  }
  return 0;
}

void backtrack(uint_fast8_t placed,
               std::span<const std::span<const uint_fast8_t[4][4]>> pieces,
               uint_fast8_t localBoard[10][10]) {
  if (placed == 8) {
    if (!valid(localBoard)) return;

    uint16_t month = get_month(localBoard);
    uint16_t day = get_day(localBoard);

    SolutionBoard currentSolution;
    for (uint_fast8_t i = 0; i < 7; ++i) {
      for (uint_fast8_t j = 0; j < 7; ++j) {
        currentSolution[i][j] = localBoard[i][j];
      }
    }

    // For verification
      std::ofstream fout(std::string("./answer/") + std::to_string(month) + '_' + std::to_string(day) + ".txt", std::ios::app);
      for (uint_fast8_t i = 0; i < 7; ++i) {
        for (uint_fast8_t j = 0; j < 7; ++j) {
          fout << static_cast<uint16_t>(currentSolution[i][j]);
        }
        fout << '\n';
      }
      fout << '\n';

    #pragma omp critical
    solutions[month - 1][day - 1].push_back(currentSolution);
    
    return;
  }

  if (placed + pieces.size() < 8)
    return;

  for (uint_fast8_t i = 0; i < pieces.size(); ++i) {
    const auto piece = pieces[i];
    for (const auto shape : piece) {
      for (uint_fast8_t x = 0; x < 7; ++x) {
        for (uint_fast8_t y = 0; y < 7; ++y) {
          if (can_place(shape, x, y, localBoard)) {
            place(shape, x, y, localBoard);
            backtrack(placed + 1, pieces.subspan(i + 1), localBoard);
            unplace(shape, x, y, localBoard);
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
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


  // start
  double start_time = CycleTimer::currentSeconds();

  uint_fast8_t temp_board_vec[48][10][10];
  uint_fast8_t cnt = 0;
  #pragma omp parallel
  {
    #pragma omp single
    {
      // first layer
      for (const auto shape : Pieces[0]) {
        for (uint_fast8_t x = 0; x < 7; ++x) {
          for (uint_fast8_t y = 0; y < 7; ++y) {
            if (can_place(shape, x, y, Board)) {
              memcpy(temp_board_vec[cnt], Board, sizeof(Board));
              place(shape, x, y, temp_board_vec[cnt]);
              ++cnt;
            }
          }
        }
      }

      // second layer
      for(const auto first_board:temp_board_vec){
        for (const auto shape : Pieces[1]) {
          for (uint_fast8_t x = 0; x < 7; ++x) {
            for (uint_fast8_t y = 0; y < 7; ++y) {
              if (can_place(shape, x, y, first_board)) {
                uint_fast8_t new_board[10][10];
                memcpy(new_board, first_board, sizeof(uint_fast8_t)*100);
                place(shape, x, y, new_board);
                #pragma omp task firstprivate(new_board)
                {
                  backtrack(2, std::span(Pieces).subspan(2), new_board);
                }
              }
            }
          }
        }
      }
      #pragma omp taskwait
    }
  }
  
  double end_time = CycleTimer::currentSeconds();
  double ElapsedTime = end_time - start_time;
  std::cout << "Elapsed Time: " << ElapsedTime << " (s)" << std::endl;

  return 0;
}

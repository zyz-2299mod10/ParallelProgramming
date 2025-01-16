# PP_final
# For one/two level method


- performance
    - baseline: 401.589 (s) , 406.027 (s) , 355.009 (s)
        - aver = 387.542 s
    - one_level:
        - thread 2: 232.321 (s) , 224.074 (s) , 235.444 (s)
            - aver: 230.613 s
            - speed up: 1.68x
        - thread 4: 92.319 (s) , 100.368 (s) , 98.6229 (s)
            - aver: 97.1033 s
            - speed up: 3.99x
    - two_level:
        - thread 2: 224.486 (s) , 225.665 (s) , 213.957 (s)
            - aver: 221.369 s
            - speed up: 1.75x
        - thread 4: 95.8378 (s) , 89.8338 (s) , 87.6937 (s)
            - aver: 91.1218 s
            - speed up: 4.25x

- run the code
    -  windows
    ```
    cd parallel
    g++ -std=c++20 -Wall -O3 -fopenmp -Iinclude -o Puzzle_solver .\one_level.cpp
    g++ -std=c++20 -Wall -O3 -fopenmp -Iinclude -o Puzzle_solver .\two_level.cpp
    ./Puzzle_solver
    ```
    - on server
    ```
    cd parallel
    make
    srun -c <num_of_core> ./Puzzle_solver_<1/2> <num_of_thread>
    ```


# For OpenMP

## Generate solution 
```
cd OpenMP
make
./Puzzle_solver [thread number: int]
```
## Visualization
> [!IMPORTANT]
> * Make sure you've generate the solutions in the `./OpenMP/result` folder
> * Python virtual environment with `flask` package
> * Remember to reload the web (F5) if u want to change the day

```
cd Visualization
python server.py
# Then Click the localhost web address 
```

**Result visualization**

<center class="half">
    <img src="https://github.com/user-attachments/assets/f426dfec-c59f-4f06-bc43-c0aa3342cbc7" width="500"/><img src="https://github.com/user-attachments/assets/bd73a0a4-3e31-4180-ba29-69a9bd6b7f41" width="500"/>
</center>


# For pthread method
- run the code
  - In lunux
    > Make sure you have 8 cores
    1. pthread with mutex
    ```
    cd pthread/mutex
    make
    ./Puzzle_solver
    ```
    2. pthread without mutex
    ```
    cd pthread/no_mutex
    make
    ./Puzzle_solver
    ```

# Reference
- [參考資料](https://github.com/ibmibmibm/a-puzzle-a-day/tree/main)
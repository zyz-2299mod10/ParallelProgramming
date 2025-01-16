document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const month = document.getElementById('month').value;
    const day = document.getElementById('day').value;
    document.getElementById('date-display').textContent = `${month}/${day}`;

    let idx = 0;
    const chessboard = document.getElementById('chessboard');

    async function fetchSolution() {
        const response = await fetch(`/solution?month=${month}&day=${day}&idx=${idx}`);
        const data = await response.json();

        if (response.ok) {
            chessboard.innerHTML = ''; // Clear previous board
            data.solution.forEach(row => {
                row.split('').forEach(num => {
                    const cell = document.createElement('div');
                    cell.classList.add('cell', `color-${num}`);
                    cell.textContent = num;
                    chessboard.appendChild(cell);
                });
            });

            document.getElementById('prev-btn').disabled = idx === 0;
            document.getElementById('next-btn').disabled = idx >= data.total - 1;
        } else {
            alert(data.error);
        }
    }

    document.getElementById('prev-btn').addEventListener('click', () => {
        if (idx > 0) {
            idx--;
            fetchSolution();
        }
    });

    document.getElementById('next-btn').addEventListener('click', () => {
        idx++;
        fetchSolution();
    });

    fetchSolution();
});
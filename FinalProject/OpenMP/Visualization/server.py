from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

RESULT_DIR = "../result"

@app.route('/')
def home():
    return send_from_directory('web', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/solution', methods=['GET'])
def get_solution():
    month = request.args.get('month')
    day = request.args.get('day')
    
    file_path = os.path.join(RESULT_DIR, f"{month}_{day}.txt")
    if not os.path.exists(file_path):
        return jsonify({"error": "Solution not found"}), 404
    
    with open(file_path, 'r') as f:
        solutions = f.read().strip().split('\n\n')
    
    idx = int(request.args.get('idx', 0))
    if idx >= len(solutions):
        return jsonify({"error": "Index out of range"}), 400
    
    return jsonify({
        "solution": solutions[idx].split('\n'),
        "total": len(solutions)
    })

if __name__ == '__main__':
    app.run(debug=True)
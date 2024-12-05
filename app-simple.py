from flask import Flask, request, jsonify

app = Flask(__name__)
app.json.ensure_ascii = False

@app.route('/echo', methods=['POST'])
def post_echo():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid input. 'message' is required."}), 400

        message = data['message']

        print(f"Received message: {message}")

        return jsonify({"received": message}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


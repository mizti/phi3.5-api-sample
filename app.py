from flask import Flask, request, jsonify
import onnxruntime_genai as og
import time

app = Flask(__name__)
app.json.ensure_ascii = False

model = None
tokenizer = None
tokenizer_stream = None
search_options = None
chat_template = None

def initialize_model():
    global model, tokenizer, tokenizer_stream, search_options, chat_template
    model_path = 'cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4'

    verbose = False
    timings = False

    if verbose:
        print("Loading model...")
    model = og.Model(f'{model_path}')
    if verbose:
        print("Model loaded")

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if verbose:
        print("Tokenizer created")

    search_options = {
        'do_sample': False,
        'max_length': 2048,
        'min_length': None,
        'top_p': None,
        'top_k': None,
        'temperature': None,
        'repetition_penalty': None
    }

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

initialize_model()

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


@app.route('/chat_completion', methods=['POST'])
def chat_completion():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid input. 'message' is required."}), 400

        message = data['message']
        print(f"Received message: {message}")

        timings = False  # 必要に応じてタイミング情報を有効化
        verbose = False
        if timings:
            started_timestamp = time.time()
            first_token_timestamp = 0
            first = True
            new_tokens = []

        prompt = f'{chat_template.format(input=message)}'
        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**{k: v for k, v in search_options.items() if v is not None})
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
        if verbose:
            print("Generator created")

        if verbose:
            print("Running generation loop...")
        if timings and first:
            first_token_timestamp = time.time()
            first = False

        response_text = ''
        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if timings and first:
                    first_token_timestamp = time.time()
                    first = False

                new_token = generator.get_next_tokens()[0]
                decoded_token = tokenizer_stream.decode(new_token)
                response_text += decoded_token
                if timings:
                    new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")

        del generator

        if timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")

        return jsonify({"response": response_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

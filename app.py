from flask import Flask, request, jsonify
from service.orchestrator import answer_user_question

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    if not user_prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    answer = answer_user_question(user_prompt)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
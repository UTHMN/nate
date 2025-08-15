from flask import Flask, request, make_response, render_template, jsonify, redirect, url_for

app = Flask(__name__)

@app.route('/')
def main() -> None:
    token = request.cookies.get('user_token')
    if token == None:
        return redirect(url_for('show_form'))
    
    data = {
        "message": "Hello, World!",
        "footer": "2025 Nate AI"
    }
    return render_template("index.html", **data)

@app.route('/login', methods=['GET'])
def show_form():
    return render_template('login.html')

@app.route('/set_token_ajax', methods=['POST'])
def set_token_ajax():
    data = request.get_json()
    token = data.get('token')

    if not token:
        return jsonify({"message": "No token provided"}), 400

    resp = make_response(jsonify({"message": "Token set successfully!"}))
    resp.set_cookie('user_token', token, httponly=True)
    return resp


if __name__ == "__main__":
    import waitress
    waitress.serve(app, host="0.0.0.0", port=8080)

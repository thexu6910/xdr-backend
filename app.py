from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "PythonAnywhere 服务已正常运行！",
        "note": "现在可以确认是你的业务代码导致了报错，我们下一步排查它"
    })

if __name__ == '__main__':
    app.run()

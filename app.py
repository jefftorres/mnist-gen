from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def generator():  # put application's code here
    return render_template('app.html')


if __name__ == '__main__':
    app.run()

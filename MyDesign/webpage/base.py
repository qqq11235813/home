from flask import Flask
from flask import render_template
from flask import request,session,redirect,flash
from sqlalchemy.orm import sessionmaker


app = Flask (__name__)


@app.route('/',methods=['GET','POST'])
def signin():
	return render_template('index.html')

@app.route('/paper')
def logout():
	return render_template('paper.html')

app.secret_key ="wow"
app.run(debug=True,port=5000)

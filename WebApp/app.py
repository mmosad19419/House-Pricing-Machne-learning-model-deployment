import os
from flask import Flask, render_template, redirect, request, session
from flask_session import Session
import pandas as pd
import numpy as np
import seaborn as sns

app = Flask(__name__)

# routes
@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main":
    app.run(debug=True)
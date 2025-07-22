{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7f751f4d-477e-4380-a2ad-30edd2171cdd",
   "metadata": {},
   "source": [
    "# Deploying Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "aca0b9f9-e9fe-44dd-8711-83baf6260553",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: flask in c:\\users\\lavan\\anaconda3\\lib\\site-packages (3.0.3)\n",
      "Requirement already satisfied: Werkzeug>=3.0.0 in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from flask) (3.0.3)\n",
      "Requirement already satisfied: Jinja2>=3.1.2 in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from flask) (3.1.4)\n",
      "Requirement already satisfied: itsdangerous>=2.1.2 in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from flask) (2.2.0)\n",
      "Requirement already satisfied: click>=8.1.3 in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from flask) (8.1.7)\n",
      "Requirement already satisfied: blinker>=1.6.2 in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from flask) (1.6.2)\n",
      "Requirement already satisfied: colorama in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from click>=8.1.3->flask) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\lavan\\anaconda3\\lib\\site-packages (from Jinja2>=3.1.2->flask) (2.1.3)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install flask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "948d8fc7-1f5d-4b2d-8fbd-f78b6787cd91",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lavan\\anaconda3\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3585: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import pandas as pd\n",
    "import joblib \n",
    "\n",
    "app = Flask(__name__)\n",
    "model = joblib.load(\"loan_rf_model.pkl\")\n",
    "scaler = joblib.load(\"loan_scaler.pkl\")\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "    data = pd.DataFrame(request.json)\n",
    "    # preprocess\n",
    "    preds = predict_loans(data)\n",
    "    return jsonify({\"predictions\": preds})\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c167fa02-c516-49e9-a616-4f0641d89c7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install gradio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b0e231c-aa5d-46f0-b213-177c5490cb9f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

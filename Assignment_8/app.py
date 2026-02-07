{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "931e1b80-57cb-45df-9a87-904866ce7fa5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "model = joblib.load('diabetes_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2e458ae2-d7cf-4066-ac47-2298f809dd9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-05 12:15:57.068 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.337 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\AKSHAT\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-01-05 12:15:57.338 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.339 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.339 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.340 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.341 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.341 Session state does not function when running a script without `streamlit run`\n",
      "2026-01-05 12:15:57.343 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.343 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.345 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.346 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.346 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.348 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.349 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.350 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.351 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.352 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.355 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:15:57.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.title('Diabetes Prediction App')\n",
    "\n",
    "glucose = st.number_input('Glucose Level',min_value=0)\n",
    "bmi = st.number_input('BMI',min_value=0.0)\n",
    "age = st.number_input('Age',min_value=0.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "097981e7-415f-47a5-8ee7-f7e8ca2d2f39",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-05 12:26:50.224 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:26:50.225 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:26:50.226 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:26:50.227 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-05 12:26:50.229 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if st.button('Predict'):\n",
    "    input_data = np.array([[glucose,bmi,age]])\n",
    "    prediction = model.predict(input_data)\n",
    "\n",
    "    if prediction [0] == 1:\n",
    "        st.error('Diabetic')\n",
    "    \n",
    "    else:\n",
    "        st.success('Non - Diabetic')\n",
    "    \n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5133ef37-a2fe-4270-a41c-cfbac63e63da",
   "metadata": {},
   "source": [
    " Button prevents auto-prediction\n",
    "\n",
    "Input converted into 2D array (required by sklearn)\n",
    "\n",
    "predict() returns 0 or 1\n",
    "\n",
    "Output shown using success() or error()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6df2316d-df1e-4bba-98de-e724f8f8ffbf",
   "metadata": {},
   "source": [
    "The trained logistic regression model was deployed using Streamlit. The model was saved using joblib and loaded into a Streamlit application. User inputs were collected through a web interface, and predictions were generated in real time. The application can be deployed locally or online using Streamlit Community Cloud.”"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b45fb364-94c0-4039-a08e-6f1416c8133b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
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
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

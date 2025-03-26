import joblib
import streamlit as st 

mb=joblib.load("spam.pkl")
cv=joblib.load("vectorizer.pkl")
# Create a title for the app
st.title("Spam Message Detector")
# Create a text input for the user to enter a message
message = st.text_area("Enter a message:")
# Create a button to submit the message
if st.button("Submit"):
    # Preprocess the message using the vectorizer
    X=cv.transform([message])
    # Use the model to predict whether the message is spam or not
    prediction = mb.predict(X)[0]
    # Display result
    if prediction == 1:
            st.error("🚨 This is SPAM!")
    else:
            st.success("✅ This is NOT SPAM.")
else:
        st.warning("⚠️ Please enter a message before checking.")
    
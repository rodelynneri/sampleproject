import streamlit as st

# Morse code dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
    'Z': '--..', 
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', 
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', 
    '8': '---..', '9': '----.', 
    ' ': '/', '.': '.-.-.-', ',': '--..--', '?': '..--..', 
    "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.', 
    ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', 
    '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-', 
    '"': '.-..-.', '$': '...-..-', '@': '.--.-.'
}

# Function to convert text to Morse code
def text_to_morse(text):
    morse_code = ' '.join(MORSE_CODE_DICT.get(char.upper(), '') for char in text)
    return morse_code

# Function to convert Morse code to text
def morse_to_text(morse_code):
    inverse_morse_dict = {v: k for k, v in MORSE_CODE_DICT.items()}
    words = morse_code.split(' / ')  # Words are separated by "/"
    decoded_message = ' '.join(
        ''.join(inverse_morse_dict.get(symbol, '') for symbol in word.split()) 
        for word in words
    )
    return decoded_message

# Streamlit app layout
st.title("Morse Code Converter 🔠➡️📡")
st.write("This tool converts text to Morse code and vice versa.")

# User input
option = st.radio("Choose Conversion Type:", ["Text to Morse Code", "Morse Code to Text"])

if option == "Text to Morse Code":
    user_input = st.text_input("Enter your text here (letters, numbers, and special characters are allowed):")
    if user_input:
        morse_code = text_to_morse(user_input)
        st.subheader("Morse Code Output:")
        st.code(morse_code, language='plaintext')
elif option == "Morse Code to Text":
    user_input = st.text_input("Enter Morse code here (use '/' to separate words):")
    if user_input:
        text_output = morse_to_text(user_input)
        st.subheader("Text Output:")
        st.code(text_output, language='plaintext')

# Footer
st.write("Made with ❤️ using Streamlit")

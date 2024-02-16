import pickle
from pathlib import Path

import streamlit_authenticator as sa

registeredNames = ['Birhan Tamiru','James Miller']

usernames = ['birhan251','james220']
passwords = ["birhan123",'james123']

encoded_pin = sa.Hasher(passwords).generate()
file_path = Path(__file__).parent / "encodede_pin.pkl"

with file_path.open("wb") as file:
    pickle.dump(encoded_pin,file)

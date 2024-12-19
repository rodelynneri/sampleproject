# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.write( GROUP MEMBERS:)

    st.write("MARY GRACE B. HERTEZ")
    st.write("BABY JEAN R. PANIZAL")
    st.write("RODELYN O. NERI")

    st.markdown(
        """
        Hello Sir ALLAN IBO our very kind instructor for modeling and simulation.
        we thank you for your gracious heart to give us this project to test and utilize our performance 
        and we would like to show you our simple project efforts and we hope you consider our work.

        And We create this project to test our capabilities and knowing the python libraries and models in applying to the streamlit app by this project we gain a hand on experience and we also import some neccessaries model along with python libraries. And on this project we creates a simple and interactive web application where users can convert text to Morse code and Morse code back to text.
        The app will have an intuitive interface where users can enter plain text or Morse code, and the output will be displayed instantly.
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        
        you can visit our pages to see our work code at the explorer of this page.



    """
    )


if __name__ == "__main__":
    run()

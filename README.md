# Sentiment Propagation
Welcome to the Sentiment Propagation GitHub repository! This project is dedicated to advancing the field of sentiment analysis through the development and distribution of innovative tools and methodologies. Our primary focus is on the exploration and enhancement of sentiment analysis techniques, particularly for underrepresented languages, by leveraging corpus-based semantic-syntactic embedding graphs and the ConGraCNet algorithm.

## Project Structure
This repository contains the core scripts that form the backbone of our sentiment propagation project:

* deprelActivation.py: Manages the activation process of dependency relations within the graph.
* deprelGraph.py: Constructs and manipulates the lexical graph based on semantic-syntactic embedding.
* deprel_Dash.py: Provides a dashboard for visual exploration of the lexical graph.
* deprel_Evaluation.py: Evaluates the performance and accuracy of the sentiment propagation algorithm.
## Getting Started
To explore the lexical graph and engage with the sentiment analysis tools we've developed, we recommend setting up a dedicated Python environment. This will ensure that all dependencies are correctly managed and that the tools run smoothly on your system.

### Step 1: Clone the Repository
Begin by cloning this repository to your local machine:

`git clone https://github.com/bperak/sentiment_propagation.git`

`cd sentiment_propagation`

### Step 2: Environment Setup
Create a virtual environment and activate it:

`python3 -m venv venv`


`source venv/bin/activate`  

On Windows use `venv\Scripts\activate`


### Install the required dependencies:

`pip install -r requirements.txt`

Step 3: Explore the Lexical Graph

### To visualize and interact with the lexical graph, run the deprel_Dash.py script:

`python deprel_Dash.py`

This will launch a local web server and provide you with a URL to access the dashboard. Open this URL in your web browser to explore the lexical graph and analyze sentiment propagation results.

### Graph Model Download
For those interested in delving deeper into the semantic-syntactic structures we've constructed, you can download the comprehensive graph model from the repository: 

[Igraph (pickled): hrWac Lexical Graph nouns](https://drive.google.com/file/d/1PyDuq47KNYjo8v-LZdIGttSMq8cAR_yx/view?usp=sharing)

This model is essential for researchers and enthusiasts looking to understand the intricate connections that underpin sentiment analysis in various languages.

## License
This project is licensed under the terms of the [LICENSE] file located in the root directory of this repository.

We invite contributors, researchers, and anyone interested in sentiment analysis to explore, modify, and build upon our work. Your feedback and contributions are highly valued as we strive to improve sentiment analysis methodologies and resources.

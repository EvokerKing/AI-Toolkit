# AI Toolkit

## Why?
I wanted to get into creating AI with no experience of how AI/ML models worked. I know plenty of Python and have been creating Discord bots with it for quite some time now.

## What models do you have and are working on adding?
Currently we have fill-mask, summarization, and text generation. At the moment, all models are using Hugging Face Transformers. The upcoming features, models, and bugfixes can be found on our Airtable databases. The database isn't public yet but will be soon.

## Who else are you doing this project with?
At first, I was the only one working with it. Now, my brother has joined to help with development.

## How does the model caching system work?
Using the built-in Python module, pickle, we are able to store objects in files. The code has automatic detection of whether the `.aimodel` files are in the folder. If not, it will auto-generate them for you.

## Why does the code not include the premade `.aimodel` files?
This is to help with safety. Pickle files can be created in such a way that they run code when loaded into a project. This can be used for viruses, malware, etc.
# Contributing to BlobCity AutoAI
If you are reading this section, then you are probably interested in contributing into this AutoAI framework. We appreciate your contribution. Thank you in advance!

We welcome you to [check the existing issues](https://github.com/blobcity/autoai/issues/) for bugs or enhancements to work on. If you have an idea for an extending features in BlobCity AutoAI, please [file a new issue](https://github.com/blobcity/autoai/issues/new) so we can discuss it.

In terms of directory structure:

* All of AutoAI's code sources are in the `blobcity` directory
* The documentation sources are in the `docs` directory

## Setup Development Environment

### How to contribute

The preferred way to contribute to BlobCity AutoAI is to fork the
[main repository](https://github.com/blobcity/autoai/) on
GitHub:

1. Fork the [project repository](https://github.com/blobcity/autoai):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourUsername/autoai.git
          $ cd autoai

3. Create a branch to hold your changes:

          $ git checkout -b my-contribution

Finally, go to the web page of your fork of the autoai repo, and click 'Pull Request' (PR) to send your changes to the maintainers for review. 

If your contribution changes autoai in any way:

* Update the [documentation](https://github.com/blobcity/autoai/tree/main/docs) so all of your changes are reflected there.

* Update the [README](https://github.com/blobcity/autoai/blob/main/README.md) if anything there has changed.

* Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.

If your contribution requires a new library dependency:

* check that the dependency is easy to install via `pip` or `conda` and are supported in Python 3. If the dependency requires a complicated installation, then we most likely won't merge your changes because we want to keep autoai easy to install.

* Add the required version of the library to requirement.txt

      

## Packaging

If you want to test your contribution on your local system performing the following steps:

1. Update the setup.cfg to account changes regarding any dependency requirement.

2. Run the following command to confirm that your contribution can be installed via pip install by navigating to our package directory :

         pip install .

## Running & Testing

run unit tests on your changes and make sure that your updated code builds and runs on Python 3

## Code Templates
Well documented code templates are maintained in a different repository: https://github.com/blobcity/ai-seed 

AI code generation takes snippers from the AI-Seed project. To make changes to the code generation output, make a contribution into the AI-Seed project. 

AI-Seed code is merged into this repository at time of library packaging, using this script: ==link required==
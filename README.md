# Deep Learning from Scratch

This repository contains the coding practice for the book [*Deep Learning from Scratch*](https://www.ituring.com.cn/book/1921). The main focus of this repository is to use unit tests extensively for the methods implemented chapter by chapter. This structure allows the repository to serve as a self-paced coding exercise for the book.

## Running Automated Tests

To run all the tests in the repository, use the following command:

```bash
pytest -s .
```

Initially, all tests will fail. As you implement more methods, more tests should pass. This approach ensures that your progress is aligned with the book's content.

To run a specific test, you can use the `-k` option to target the desired test. For example:

```bash
pytest -s ch02_perceptron/test_perceptron.py -k test_AND
```

This command runs the `test_AND` test in the `test_perceptron.py` file located in the `ch02_perceptron` directory.

## Code Quality and Checking

To maintain high code quality and consistency, this repository integrates several tools with pre-commit hooks.
**You can ignore this part if you don't do it.**

### Setting Up Pre-Commit Hooks

To set up the pre-commit hooks, run the following command:

```bash
pre-commit install
```

This installs the pre-commit hooks defined in the `.pre-commit-config.yaml` file. Once installed, these hooks will automatically check your code for any issues every time you make a commit.

### Running Pre-Commit Hooks Manually

You can also manually run the pre-commit hooks on all files using the following command:

```bash
pre-commit run --all-files
```

This setup helps ensure that your code adheres to defined standards and maintains high quality throughout the development process.

---

Hope this repo can help you studying the deep learning.

# SPADE_score

SPADE (Spectral Method for Black-Box Adversarial Robustness Evaluation) is a spectral method used for assessing the robustness of adversarial AI models in a black-box environment.

## Requirements
To use SPADE, the following dependencies are required:
- `hnswlib`
- `Julia`
- `NetworkX`
- `Pyjulia`

## Usage
### Evaluating SPADE-Score

To evaluate the SPADE-Score, follow the steps outlined below:

1. Download all the necessary files.

2. In your Python code, import the SPADE module:

    ```python
    from SPADE import Spade
    ```

3. Use the Spade function with your data inputs and outputs:

    ```python
    TopEig, TopEdgeList, TopNodeList, node_score, L_in, L_out, Dxy, Uxy = Spade(data_input, data_output)
    ```

4. The `data_input` and `data_output` parameters should be flattened. If your data is a multidimensional array, make sure to flatten it before use.

5. The `Spade` function has the following default options:
   - `k=10`: Specifies the kNN graph.
   - `num_eigs=2`: Determines the number of general eigenpairs.
   - `sparse=False`: Indicates whether to construct a sparse kNN graph.
   - `weighted=False`: Determines whether to construct a weighted graph.

### Troubleshooting
In certain situations, the `scipy` might not accurately compute general eigenvalues, leading to a scenario where the maximum eigenvalues are negative. To circumvent this issue, we utilize a function from Julia to calculate the general eigenpairs.




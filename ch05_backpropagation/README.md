# Float Precision

Starting from this chapter, we should use the:
- **np_float**
- **np_array**
- and other functions in **common/default_type_array.py**

to generate arrays and other numbers.
This allows us to set the float precision
since NumPy does not support a set_default_type function. The default float if np.float32.

### Mixed Precision Training

For large models, mixed precision training is a mainstream
approach, typically combining FP16 and FP32:
- **Forward and Backward Propagation**: Use FP16 to accelerate computation and
reduce memory usage.
- **Weight Updates**: Maintain the main weights in FP32 to avoid cumulative
errors caused by low precision.
- **Gradient Scaling**: Scale up the gradients of the loss function to prevent
underflow in FP16, then scale them back to the original value during updates.

**Unlike GPUs, CPUs may have limited support for float16**, so the actual float16
acceleration effect may not be significant.

# Place your trained PyTorch model here

Place your `model.pt` file in this directory.

The model should accept input tensors of shape `(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)` and output logits for emotion classification.

## Expected Model Structure

Your model should:

- Accept RGB images of size 224x224 (configurable in .env)
- Output logits for each emotion class
- Support the emotions defined in your .env file

## Example Model Output

```python
# Input shape: (1, 3, 224, 224)
# Output shape: (1, num_emotions)
```

If no model is present, the service will use a placeholder model that returns random predictions for testing purposes.

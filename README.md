# Diffusion-model-from-scratch-using-pytorch
![alt text](<Model atchitecture.JPG>)

The image demonstrates the architecture of the diffusion model. It is consist of the following part each one is coded in a different python file in the sd folder:

1.Random Noise: This is the starting point of the model where a noise distribution is generated. This random noise acts as the initial "canvas" which the model will iteratively refine into a coherent image.


2.Encoder: This component transforms the initial noise into a latent representation (denoted as Z). The encoder compresses the noise into a more structured format that the diffusion model can manipulate over several steps.
Text Prompt: This is the input text description that specifies what the generated image should depict (e.g., "a dog with glasses"). The text is converted into data that the model can understand.


3.CLIP Encoder: The CLIP (Contrastive Language–Image Pre-training) Encoder processes the text prompt and converts it into embeddings. These embeddings are a numerical representation of the text prompt, which allows the model to understand and retain the semantic meaning of the input text.
Prompt Embeddings: The output of the CLIP Encoder. These embeddings effectively communicate the characteristics and attributes of the image as described in the text prompt to the model.


4.Scheduler: This manages the progression of the model through its steps, typically using a concept known as 'time embedding'. It controls how the diffusion process (the transformation of noise into a detailed image) progresses over time, deciding when and how the model's parameters should adjust at each step.


5.Time Embeddings: These are specific parameters that inform the model of the current step in the diffusion process. They help the model adjust its behavior based on how far along it is in the process of refining the noise into an image.


6.Intermediate Model States: Represented by the series of blue and gray blocks, these show how the model iteratively refines the noise into an image. Each block represents a step in the diffusion process where the model slightly reduces the randomness (noise) and increases the image details according to the prompt embeddings and time embeddings.


7.Decoder: After the diffusion steps (denoted as T steps), the final latent representation (Z') is passed to the decoder. The decoder transforms this latent representation back into an image space, resulting in the final image (X') that corresponds to the text prompt.


8.Final Image (X'): This is the output of the model, a generated image that visually represents the text prompt, synthesized from the iterative refinement of noise.